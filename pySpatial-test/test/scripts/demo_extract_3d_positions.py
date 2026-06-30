#!/usr/bin/env python
"""CLI demo for object_3d_extraction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_TEST_DIR = Path(__file__).resolve().parents[1]
if str(REPO_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_TEST_DIR))

from object_3d_extraction import Object3DExtractionConfig, Object3DLocator
from object_3d_extraction.prompts import (
    PATTERN_GET_OBJECTS_OF_INTEREST,
    PROMPT_GET_OBJECTS_OF_INTEREST,
    PROMPT_GET_OBJECTS_OF_INTEREST_AUX,
    QUESTION_TYPE_RULES,
)
from object_3d_extraction.object_3d_locator import is_count_ratio_question
from object_3d_extraction.utils import extract_object_names_from_question_options

DEFAULT_LOCAL_QWEN_MODEL_PATH = "/data/pretrain_models/Qwen/models--Qwen--Qwen2.5-VL-7B-Instruct"


class QwenVLRefinementModel:
    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoProcessor

        model_cls = _resolve_qwen_vl_model_class()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else device,
            trust_remote_code=True,
        )

    def process_messages(self, messages, max_new_tokens=32, do_sample=False, temperature=0.0):
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]



class OpenAIVLRefinementModel:
    def __init__(self, api_key: str = None, model: str = "gpt-4.1", base_url: str = None):
        import os
        from openai import OpenAI

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("CLOSEAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key.")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def process_messages(self, messages, max_new_tokens=32):
        import base64
        from io import BytesIO

        def image_to_url(image):
            buf = BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return "data:image/png;base64," + b64

        content = []
        for item in messages[0].get("content", []):
            if item.get("type") == "text":
                content.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                content.append({"type": "input_image", "image_url": image_to_url(item.get("image"))})
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_new_tokens,
        )
        return response.output_text.strip()


def _resolve_qwen_vl_model_class():
    import transformers

    candidates = [
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
    ]
    for name in candidates:
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
    raise ImportError(
        "No compatible Qwen-VL model class found in transformers. "
        "Please install a transformers version that supports Qwen2.5-VL or AutoModelForVision2Seq."
    )

def build_vlm_refinement_model(args):
    if not args.use_vlm_refinement and not getattr(args, "use_vlm_object_extraction", True):
        return None
    if getattr(args, "backend", "local_qwen") == "openai":
        print("Loading OpenAI refinement model {}...".format(args.api_model))
        return OpenAIVLRefinementModel(api_key=args.api_key, model=args.api_model, base_url=getattr(args, "base_url", None))
    print("Loading Qwen2.5-VL refinement model from {}...".format(args.vlm_model_path))
    return QwenVLRefinementModel(args.vlm_model_path, device=args.device)


class ObjectExtractionRunner:
    def __init__(
        self,
        device: str = "cuda",
        mask_fallback: str = "auto",
        use_vlm_refinement: bool = True,
        vlm_model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH,
        vlm_model=None,
        backend: str = "local_qwen",
        api_model: str = "gpt-4.1",
        api_key: str = None,
        base_url: str = None,
        use_vlm_object_extraction: bool = True,
        count_max_instances: int = 20,
    ):
        print("Initializing reusable object extractor...")
        self.device = device
        self.mask_fallback = mask_fallback
        self.args = argparse.Namespace(
            image=None,
            image_root="/data/datasets/Omni3D-Bench/images",
            base_data_path=None,
            device=device,
            box_threshold=0.05,
            text_threshold=0.05,
            use_vlm_refinement=use_vlm_refinement,
            use_vlm_object_extraction=use_vlm_object_extraction,
            count_max_instances=count_max_instances,
            vlm_model_path=vlm_model_path,
            mask_fallback=mask_fallback,
        )
        config = Object3DExtractionConfig()
        config.detection.box_threshold = self.args.box_threshold
        config.detection.text_threshold = self.args.text_threshold
        config.detection.use_vlm_refinement = use_vlm_refinement
        config.detection.count_max_instances = count_max_instances
        config.mask_fallback_mode = mask_fallback

        if vlm_model is None and (use_vlm_refinement or use_vlm_object_extraction):
            if backend == "openai":
                vlm_model = OpenAIVLRefinementModel(api_key=api_key, model=api_model, base_url=base_url)
            else:
                vlm_model = QwenVLRefinementModel(vlm_model_path, device=device)

        self.vlm_model = vlm_model
        self.locator = Object3DLocator(config=config, device=device, vlm_model=vlm_model)

    def extract_sample(
        self,
        sample: dict,
        output_root,
        visualize: bool = True,
        image: str = None,
        base_data_path: str = None,
    ):
        args = argparse.Namespace(**vars(self.args))
        args.image = image
        args.base_data_path = base_data_path
        sample_key = get_sample_key(sample, 0)
        resolved_image = resolve_image_path(args, sample)
        object_info = resolve_object_names(
            sample,
            image=resolved_image,
            vlm_model=self.vlm_model if self.args.use_vlm_object_extraction else None,
            use_vlm_object_extraction=self.args.use_vlm_object_extraction,
        )
        object_names = object_info["objects"]
        sample_save_dir = Path(output_root) / sample_key
        result = self.locator.extract(
            image=resolved_image,
            object_names=object_names,
            visualize=visualize,
            save_dir=sample_save_dir,
            question=sample.get("question", ""),
            answer=sample.get("answer", sample.get("gt_answer", "")),
            question_type=object_info["question_type"],
            object_extraction_items=object_info.get("object_extraction_items"),
        )
        record = {
            "sample_id": sample_key,
            "image": resolved_image,
            "objects": object_names,
            "object_extraction_method": object_info["method"],
            "vlm_extracted_objects": object_info["vlm_extracted_objects"],
            "rule_extracted_objects": object_info["rule_extracted_objects"],
            "object_extraction_response": object_info["object_extraction_response"],
            "object_extraction_items": object_info.get("object_extraction_items", []),
            "question_type": object_info["question_type"],
            "result": result,
        }
        output_path = write_sample_json(sample_save_dir, sample_key, record)
        return record, str(output_path)


def extract_objects_for_sample(
    sample: dict,
    output_root,
    device: str = "cuda",
    mask_fallback: str = "auto",
    use_vlm_refinement: bool = True,
    vlm_model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH,
    vlm_model=None,
    visualize: bool = True,
    image: str = None,
    base_data_path: str = None,
    backend: str = "local_qwen",
    api_model: str = "gpt-4.1",
    api_key: str = None,
    base_url: str = None,
    extractor=None,
    use_vlm_object_extraction: bool = True,
):
    if extractor is not None:
        return extractor.extract_sample(
            sample,
            output_root=output_root,
            visualize=visualize,
            image=image,
            base_data_path=base_data_path,
        )

    runner = ObjectExtractionRunner(
        device=device,
        mask_fallback=mask_fallback,
        use_vlm_refinement=use_vlm_refinement,
        vlm_model_path=vlm_model_path,
        vlm_model=vlm_model,
        backend=backend,
        api_model=api_model,
        api_key=api_key,
        base_url=base_url,
        use_vlm_object_extraction=use_vlm_object_extraction,
    )
    return runner.extract_sample(
        sample,
        output_root=output_root,
        visualize=visualize,
        image=image,
        base_data_path=base_data_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 3D object positions from Omni3D-Bench samples.")
    parser.add_argument("--image", default=None, help="Optional image path override.")
    parser.add_argument(
        "--dataset_json",
        default="/data/datasets/Omni3D-Bench/annotations.json",
        help="Path to Omni3D-Bench annotations JSON.",
    )
    parser.add_argument(
        "--image_root",
        default="/data/datasets/Omni3D-Bench/images",
        help="Root directory for Omni3D-Bench image_filename paths.",
    )
    parser.add_argument("--sample_json", default=None, help="Path to one JSON sample file.")
    parser.add_argument("--jsonl", default=None, help="Path to a MindCube JSONL file.")
    parser.add_argument("--sample_id", default=None, help="Sample id to select from --jsonl.")
    parser.add_argument("--sample_index", type=int, default=None, help="Single sample index to run.")
    parser.add_argument("--sample_start_index", type=int, default=None, help="Inclusive start index for a sample range.")
    parser.add_argument("--sample_end_index", type=int, default=None, help="Inclusive end index for a sample range.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to run.")
    parser.add_argument(
        "--base_data_path",
        default=None,
        help="Base path for relative sample image paths when --image is not provided.",
    )
    parser.add_argument("--device", default="cuda", help="Device for model inference, e.g. cuda or cpu.")
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.05,
        help="GroundingDINO box threshold. Omni3D-Bench uses APC-VLM-style low-threshold recall.",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.05,
        help="GroundingDINO text threshold. Omni3D-Bench uses APC-VLM-style low-threshold recall.",
    )
    parser.add_argument(
        "--use_vlm_refinement",
        action="store_true",
        default=True,
        help="Enable APC-VLM-style Qwen-VL refinement.",
    )
    parser.add_argument(
        "--vlm_model_path",
        default=DEFAULT_LOCAL_QWEN_MODEL_PATH,
        help="Local Qwen2.5-VL model path for detection candidate refinement.",
    )
    parser.add_argument("--backend", choices=("local_qwen", "openai"), default="local_qwen", help="VLM refinement backend.")
    parser.add_argument("--api_model", default="gpt-4.1", help="OpenAI model for VLM refinement when --backend openai.")
    parser.add_argument("--api_key", default=None, help="OpenAI API key for VLM refinement; falls back to OPENAI_API_KEY.")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible base URL; falls back to CLOSEAI_BASE_URL or OPENAI_BASE_URL.")
    parser.add_argument(
        "--no_vlm_refinement",
        action="store_false",
        dest="use_vlm_refinement",
        help="Disable VLM refinement and use rule-based candidate ranking.",
    )
    parser.add_argument(
        "--use_vlm_object_extraction",
        action="store_true",
        default=True,
        help="Enable APC-VLM-style VLM extraction of objects of interest.",
    )
    parser.add_argument(
        "--no_vlm_object_extraction",
        action="store_false",
        dest="use_vlm_object_extraction",
        help="Disable VLM object extraction and use rule-based question parsing.",
    )
    parser.add_argument(
        "--mask_fallback",
        choices=("auto", "off"),
        default="auto",
        help="Control entity mask fallback. auto enables fallback; off always uses SAM mask for 3D.",
    )
    parser.add_argument(
        "--save_dir",
        default="outputs/object_3d_extraction",
        help="Root directory for per-sample JSON outputs and debug visualizations.",
    )
    parser.add_argument("--no_visualize", action="store_true", help="Disable debug visualization output.")
    parser.add_argument(
        "--count_max_instances",
        type=int,
        default=20,
        help="Maximum number of indexed instances to keep for counting/count-ratio targets.",
    )
    return parser.parse_args()


def load_samples(args: argparse.Namespace):
    if args.sample_json:
        with open(args.sample_json, "r", encoding="utf-8") as f:
            return [json.load(f)]

    if args.jsonl:
        all_samples = []
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))

        if args.sample_id is not None:
            for sample in all_samples:
                if sample.get("id") == args.sample_id:
                    return [sample]
            raise ValueError("sample_id not found in jsonl: {}".format(args.sample_id))

        return _select_samples_by_index_args(all_samples, args)

    with open(args.dataset_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset.get("questions", [])
    if args.sample_id is not None:
        for sample in samples:
            if get_sample_key(sample, 0) == args.sample_id:
                return [sample]
        raise ValueError("sample_id not found in dataset_json: {}".format(args.sample_id))

    return _select_samples_by_index_args(samples, args)


def _select_samples_by_index_args(samples, args):
    if args.sample_index is not None and (args.sample_start_index is not None or args.sample_end_index is not None):
        raise ValueError("Use either --sample_index or --sample_start_index/--sample_end_index, not both")

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise ValueError("sample_index out of range: {}".format(args.sample_index))
        return [samples[args.sample_index]]

    if args.sample_start_index is not None or args.sample_end_index is not None:
        start = args.sample_start_index if args.sample_start_index is not None else 0
        end = args.sample_end_index if args.sample_end_index is not None else len(samples) - 1
        if start < 0 or end < start or start >= len(samples):
            raise ValueError("sample index range out of range: {}-{}".format(start, end))
        selected = samples[start : min(end, len(samples) - 1) + 1]
        if args.max_samples is not None:
            selected = selected[: args.max_samples]
        return selected

    if args.max_samples is not None:
        return samples[: args.max_samples]
    return samples


def resolve_image_path(args: argparse.Namespace, sample) -> str:
    if args.image:
        return args.image

    image_filename = sample.get("image_filename")
    if image_filename:
        image_path = Path(image_filename)
        if image_path.is_absolute():
            return str(image_path)
        return str(Path(args.image_root) / image_path)

    images = sample.get("images") or []
    if not images:
        raise ValueError("Sample does not contain images; please pass --image")

    image_path = Path(images[0])
    if image_path.is_absolute():
        return str(image_path)

    if not args.base_data_path:
        raise ValueError("Sample image path is relative; please pass --base_data_path")

    return str(Path(args.base_data_path) / image_path)


def get_sample_key(sample, fallback_index: int) -> str:
    if "question_index" in sample:
        return "omni3d_{}".format(sample["question_index"])
    return sample.get("id", "sample_{}".format(fallback_index))


COUNT_QUESTION_RE = re.compile(r"\b(how many|number of|count|total number)\b", re.IGNORECASE)
NON_VISUAL_HOW_MANY_RE = re.compile(
    r"\b(?:stack|stacked|reach|match|same height|fit|volume|width|height|length|depth)\b",
    re.IGNORECASE,
)
OPTION_QUESTION_RE = re.compile(
    r"\b(options?\s*:|choose from|one of|which of|which group|which object|which one|which is|what object|what is closer|what is farther|what is further|what is nearest|what is closest|what is furthest|what is farthest)",
    re.IGNORECASE,
)
YES_NO_QUESTION_RE = re.compile(
    r"^\s*(?:is|are|was|were|will|would|can|could|do|does|did|has|have|should)\b",
    re.IGNORECASE,
)
YES_NO_ANSWER_RE = re.compile(r"^\s*(yes|no)\s*[.!]?\s*$", re.IGNORECASE)
SAME_TYPE_EXISTENCE_RE = re.compile(
    r"\b(?:two|2|multiple|more than one|same)\b.*\b(?:same object types?|same objects?|object types?)\b|"
    r"\b(?:are|is) there (?:two|2|multiple|more than one) of the same objects?\b",
    re.IGNORECASE,
)


def is_same_type_existence_question(question: str) -> bool:
    return bool(SAME_TYPE_EXISTENCE_RE.search(str(question or "")))


def _sample_answer(sample):
    for key in ("answer", "expected_answer", "gt_answer"):
        if key in sample and sample.get(key) is not None:
            return sample.get(key)
    return None


def _is_yes_no_answer(answer) -> bool:
    return bool(YES_NO_ANSWER_RE.match(str(answer or "")))


def classify_primary_question_type(sample) -> str:
    question = sample.get("question", "")
    answer = _sample_answer(sample)

    if _is_yes_no_answer(answer) or YES_NO_QUESTION_RE.search(question):
        return "yes_no"
    if OPTION_QUESTION_RE.search(question):
        return "multi_choice"
    return "number"


def is_number_counting_question(question: str, answer_type: str = None) -> bool:
    answer_type = str(answer_type or "").lower()
    if is_count_ratio_question(question):
        return True
    if answer_type == "int":
        return True
    if COUNT_QUESTION_RE.search(question) and not NON_VISUAL_HOW_MANY_RE.search(question):
        return True
    return False


def classify_question_type(sample) -> str:
    question = sample.get("question", "")
    answer_type = str(sample.get("answer_type") or "").lower()
    primary_type = classify_primary_question_type(sample)

    if primary_type != "number":
        return primary_type
    if is_number_counting_question(question, answer_type):
        return "number_ct"
    return "number_other"


def build_object_extraction_prompt(question: str, question_type: str = "generic") -> str:
    rules = QUESTION_TYPE_RULES.get(question_type, QUESTION_TYPE_RULES["generic"])
    return PROMPT_GET_OBJECTS_OF_INTEREST.format(
        question=question,
        question_type_rules=rules.strip(),
    )


def _normalize_object_phrase_for_merge(name: str) -> str:
    text = re.sub(r"[_-]+", " ", str(name or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"\b(rightmost|leftmost|topmost|bottommost|right most|left most|top most|bottom most|right|left|middle|center|centered|nearest|closest|furthest|farthest|front|back)\b",
        " ",
        text,
    )
    text = re.split(r"\b(under|below|above|over|next to|near|beside|to the right of|to the left of|right of|left of|in front of|behind|inside|on top of|on)\b", text)[0]
    text = re.sub(
        r"\b(white|black|gray|grey|brown|red|blue|green|yellow|orange|purple|pink|wooden|wood|metal|metallic|glass|transparent|translucent|clear|circular|round|square|striped|solid|polka dot|polka-dot|small|large|big|tall|short)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split():
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


INSTANCE_MODIFIER_PATTERN = re.compile(
    r"\b("
    r"rightmost|leftmost|topmost|bottommost|right-most|left-most|top-most|bottom-most|right most|left most|top most|bottom most|"
    r"center|centered|middle|upper|lower|nearest|closest|furthest|farthest|left|right|front|back|"
    r"white|black|gray|grey|brown|red|blue|green|yellow|glass|wooden|metal|transparent|translucent|clear|circular|round|striped|solid|polka[- ]dot|"
    r"under|below|above|next to|beside|right of|left of|in front of|behind|inside|on top of"
    r")\b",
    re.IGNORECASE,
)


def _instance_modifier_signature(name: str) -> tuple:
    text = str(name or "").lower()
    normalized = []
    for modifier in INSTANCE_MODIFIER_PATTERN.findall(text):
        modifier = modifier.replace("-", " ").strip()
        if modifier == "centered":
            modifier = "center"
        if modifier in {"right most", "rightmost"}:
            modifier = "rightmost"
        elif modifier in {"left most", "leftmost"}:
            modifier = "leftmost"
        elif modifier in {"top most", "topmost"}:
            modifier = "topmost"
        elif modifier in {"bottom most", "bottommost"}:
            modifier = "bottommost"
        elif modifier == "grey":
            modifier = "gray"
        elif modifier == "farthest":
            modifier = "furthest"
        normalized.append(modifier)
    return tuple(sorted(set(normalized)))


def _has_precise_instance_modifier(name: str) -> bool:
    return bool(_instance_modifier_signature(name))


def _same_category_but_distinct_instance(a: str, b: str) -> bool:
    if not _same_general_object_category(a, b):
        return False
    sig_a = _instance_modifier_signature(a)
    sig_b = _instance_modifier_signature(b)
    return bool(sig_a and sig_b and sig_a != sig_b)


def _same_general_object_category(a: str, b: str) -> bool:
    base_a = _normalize_object_phrase_for_merge(a)
    base_b = _normalize_object_phrase_for_merge(b)
    if not base_a or not base_b:
        return False
    return base_a == base_b or base_a in base_b.split() or base_b in base_a.split()


def _is_distinct_compound_object(a: str, b: str) -> bool:
    left = re.sub(r"\s+", " ", str(a or "").lower().replace("television", "tv")).strip()
    right = re.sub(r"\s+", " ", str(b or "").lower().replace("television", "tv")).strip()
    pair = {left, right}
    if pair == {"tv", "tv stand"}:
        return True
    return False


BARE_ATTRIBUTE_WORDS = {
    "white", "black", "gray", "grey", "brown", "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "wooden", "wood", "metal", "metallic", "glass", "transparent", "translucent", "clear", "leather",
}


def _is_bare_attribute_phrase(name: str) -> bool:
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    return text in BARE_ATTRIBUTE_WORDS


def _has_attribute_category_phrase(name: str) -> bool:
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    return len(text.split()) > 1 and any(re.search(r"\b{}\b".format(re.escape(attr)), text) for attr in BARE_ATTRIBUTE_WORDS)


def postprocess_extracted_objects(objects: list, question: str = "", question_type: str = "generic", split_relation_phrases: bool = True) -> list:
    output = []
    for obj in objects or []:
        items = _split_synthetic_object_phrase(obj, question_type=question_type) if split_relation_phrases else [obj]
        for item in items:
            cleaned = _clean_extracted_object_phrase(item, question)
            if cleaned and not _looks_like_non_object_phrase(cleaned):
                output.append(cleaned)
    if is_same_type_existence_question(question):
        output = [obj for obj in output if not re.search(r"\b(?:same object types?|same objects?|object types?)\b", obj)]
    return _dedupe_preserve_order(output)



def _object_phrase_explicitly_mentioned(question: str, object_name: str) -> bool:
    question_text = re.sub(r"[_-]+", " ", str(question or "").lower())
    object_text = re.sub(r"[_-]+", " ", str(object_name or "").lower())
    object_text = re.sub(r"\s+", " ", object_text).strip()
    if not object_text:
        return False
    if object_text in question_text:
        return True
    singular = _normalize_object_phrase_for_merge(object_text)
    return bool(singular and re.search(r"\b{}s?\b".format(re.escape(singular)), question_text))


def _should_keep_both_objects(question: str, a: str, b: str) -> bool:
    if (_is_bare_attribute_phrase(a) and _has_attribute_category_phrase(b)) or (_is_bare_attribute_phrase(b) and _has_attribute_category_phrase(a)):
        return False
    if _is_distinct_compound_object(a, b):
        return True
    if not _same_general_object_category(a, b):
        return True
    sig_a = _instance_modifier_signature(a)
    sig_b = _instance_modifier_signature(b)
    if sig_a and sig_b:
        return sig_a != sig_b or (
            _object_phrase_explicitly_mentioned(question, a)
            and _object_phrase_explicitly_mentioned(question, b)
            and a != b
        )
    # One precise relation/attribute/position phrase and one generic category: keep only the precise one.
    if bool(sig_a) != bool(sig_b):
        return False
    # Both generic same-category phrases are duplicates.
    return False


def _clean_extracted_object_phrase(name: str, question: str = "") -> str:
    text = re.sub(r"\s+", " ", str(name or "").lower()).strip(" \"'`({[])}.,;:!?")
    text = text.replace("left-most", "leftmost").replace("right-most", "rightmost").replace("top-most", "topmost")
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"\bcombined$", "", text).strip()
    text = re.sub(r"^(?:combined|same|double)\s+", "", text).strip()
    text = re.sub(r"\b(?:closer|farther|further|nearer|nearest|closest|furthest|farthest|next)$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    if text == "camera" and not re.search(r"\b(?:visible|physical|security|digital) camera\b", str(question or "").lower()):
        return ""
    if re.search(r"\b(?:same object types?|same objects?|object type|object types|physical objects|required objects)\b", text):
        return ""
    return text


RELATION_CONNECTOR_RE = re.compile(
    r"\b(to the left of|to the right of|in front of|on top of|at the end of|next to|right of|left of|beside|under|below|above|over|behind|inside|near)\b",
    re.IGNORECASE,
)


def _split_synthetic_object_phrase(name: str, question_type: str = "generic") -> list:
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    text = text.replace("television", "tv")
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    text = re.sub(r"\bcombined\b", "", text).strip()
    if " and " in text and re.search(r"\b(tv|stand|table|chair|sink|cabinet|sofa|bed|dresser|nightstand|remote|coaster)\b", text):
        parts = [part.strip() for part in text.split(" and ") if part.strip()]
        if len(parts) == 2:
            left, right = parts
            left = _clean_extracted_object_phrase(left)
            right = _clean_extracted_object_phrase(right)
            if left and right:
                return [left, right]
    if question_type in {"number_other", "numeric_other", "multi_choice", "choice_object"} and RELATION_CONNECTOR_RE.search(text):
        return [name]
    relation_match = re.search(r"\b(.+?)\s+(?:to the left of|to the right of|in front of|on top of|next to|left of|right of|beside|under|below|above|over|behind|inside|near)\s+(?:the |a |an )?(.+)$", text)
    if relation_match:
        left = _clean_extracted_object_phrase(relation_match.group(1), question="")
        right = _clean_extracted_object_phrase(relation_match.group(2), question="")
        if left and right:
            return [left, right]

    return [name]


def merge_vlm_and_rule_objects(question_type: str, vlm_objects: list, rule_objects: list, question: str = "") -> list:
    if not vlm_objects or not rule_objects:
        return vlm_objects or rule_objects

    def is_duplicate(candidate: str, existing: str) -> bool:
        return _same_general_object_category(candidate, existing) and not _should_keep_both_objects(question, candidate, existing)

    merged = []
    for obj in rule_objects:
        if _has_precise_instance_modifier(obj) and obj not in merged:
            merged.append(obj)

    for obj in vlm_objects:
        if not any(is_duplicate(obj, current) for current in merged) and obj not in merged:
            merged.append(obj)

    for obj in rule_objects:
        if not any(is_duplicate(obj, current) for current in merged) and obj not in merged:
            merged.append(obj)

    return merged or vlm_objects


def resolve_object_names(
    sample,
    image: str = None,
    vlm_model=None,
    use_vlm_object_extraction: bool = True,
) -> dict:
    question = sample.get("question", "")
    question_type = classify_question_type(sample)
    split_rule_relation_phrases = question_type in {"yes_no", "number_ct", "number_vt", "numeric_ct"}
    rule_objects = postprocess_extracted_objects(
        resolve_object_names_by_rule(sample),
        question,
        question_type,
        split_relation_phrases=split_rule_relation_phrases,
    )
    vlm_objects = []
    vlm_response = None

    if use_vlm_object_extraction and vlm_model is not None and image is not None:
        try:
            vlm_objects, vlm_response, vlm_items = extract_object_names_with_vlm(image, question, vlm_model, question_type=question_type)
            if not vlm_items:
                vlm_objects = postprocess_extracted_objects(vlm_objects, question, question_type)
                vlm_items = make_object_extraction_items(vlm_objects)
        except Exception as exc:
            vlm_response = "ERROR: {}".format(exc)
            vlm_items = []

    if vlm_objects:
        return {
            "objects": vlm_objects,
            "method": "vlm",
            "vlm_extracted_objects": vlm_objects,
            "rule_extracted_objects": rule_objects,
            "object_extraction_response": vlm_response,
            "object_extraction_items": vlm_items,
            "question_type": question_type,
        }

    if rule_objects:
        return {
            "objects": rule_objects,
            "method": "rule_fallback" if use_vlm_object_extraction else "rule",
            "vlm_extracted_objects": vlm_objects,
            "rule_extracted_objects": rule_objects,
            "object_extraction_response": vlm_response,
            "object_extraction_items": make_object_extraction_items(rule_objects),
            "question_type": question_type,
        }

    raise ValueError("Could not extract object names from sample question")


def resolve_object_names_by_rule(sample) -> list:
    question = sample.get("question", "")
    object_names = extract_object_names_from_omni3d_question(question)
    if not object_names:
        object_names = extract_object_names_from_question_options(question)
    object_names = [name for name in object_names if not _looks_like_non_object_phrase(name)]
    return _dedupe_preserve_order(object_names)

def _extract_relation_operand_phrases(question: str) -> list:
    names = []
    relation_words = r"to the right of|to the left of|left of|right of|next to|beside|under|below|above|over|in front of|behind|inside|on top of|near"
    pattern = re.compile(
        r"\b(?:the|a|an)\s+((?:combined\s+)?[a-z0-9 -]+?\s+(?:" + relation_words + r")\s+(?:the|a|an)\s+[a-z0-9 -]+?)(?=\s+(?:is|are|was|were|would|will|could|should|in|with|than|or|and|as)\b|[?,.]|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.findall(question):
        cleaned = _clean_omni3d_object_name(match)
        if cleaned and not _looks_like_non_object_phrase(cleaned):
            names.append(cleaned)
    return names

def _extract_relation_reference_object_names(question: str) -> list:
    names = []
    relation_pattern = re.compile(
        r"\b(?:to the right of|to the left of|in front of|on top of|next to|right of|left of|to|than|from|under|below|above|over|behind|inside|near|beside|of)\s+(?:the|a|an)\s+(.+?)(?=\s+(?:is|are|was|were|would|will|could|should|to|of|on|in|under|above|below|behind|from|with|towards|than|or|and|as|directly|right|left|front|back)\b|[?,.]|$)",
        flags=re.IGNORECASE,
    )
    for match in relation_pattern.findall(question):
        cleaned = _clean_omni3d_object_name(match)
        if cleaned and not _looks_like_non_object_phrase(cleaned):
            names.append(cleaned)
    return names


def _extract_distribution_count_object_names(question: str) -> list:
    names = []
    text = re.sub(r"\s+", " ", str(question or "").lower())
    match = re.search(
        r"\benough\s+(.+?)\s+in\s+(?:the|a|an)\s+(.+?)\s+for\s+each\s+(.+?)(?:\s+at\s+(?:the|a|an)\s+(.+?))?\s+to\s+get\s+one\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return names
    for part in match.groups():
        if not part:
            continue
        cleaned = _clean_omni3d_object_name(part)
        if cleaned == "person":
            cleaned = "people"
        if cleaned and not _looks_like_non_object_phrase(cleaned):
            names.append(cleaned)
    return names



def extract_object_names_with_vlm(image, question: str, vlm_model, num_tries: int = 2, question_type: str = "generic") -> tuple:
    from PIL import Image

    image_pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    response = None
    for try_idx in range(num_tries):
        if try_idx == 0:
            prompt = build_object_extraction_prompt(question, question_type)
        else:
            prompt = PROMPT_GET_OBJECTS_OF_INTEREST_AUX.format(question=question, response=response or "")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil.resize((400, 400))},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = vlm_model.process_messages(messages, max_new_tokens=128)
        objects, object_items = parse_vlm_object_extraction_response(
            response,
            preserve_duplicates=is_same_type_existence_question(question),
        )
        if objects:
            return objects, response, object_items
    return [], response, []


def parse_vlm_object_names(response: str, preserve_duplicates: bool = False) -> list:
    text = str(response or "")
    detect_match = re.search(r"\[Detect\]\s*(\[[^\]]*\])", text, flags=re.IGNORECASE)
    matches = [detect_match.group(1)] if detect_match else re.findall(PATTERN_GET_OBJECTS_OF_INTEREST, text)
    if not matches:
        return []
    match = matches[0] if detect_match else matches[-1]
    names = []
    for part in match.strip().replace("[", "").replace("]", "").split(","):
        cleaned = part.strip().lower().replace("'", "").replace('\"', "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and not _looks_like_non_object_phrase(cleaned):
            names.append(cleaned)
    return names if preserve_duplicates else _dedupe_preserve_order(names)


def make_object_extraction_items(objects: list) -> list:
    return [
        {"detect_phrase": obj, "object": obj, "relation_context": "", "reference_object": ""}
        for obj in _dedupe_preserve_order([str(item or "").strip().lower() for item in objects])
        if obj
    ]


def _is_camera_reference(text: str) -> bool:
    return str(text or "").strip().lower() in {"camera", "image", "image viewpoint", "viewpoint"}


def _normalize_object_extraction_item(item, fallback_phrase: str = "") -> dict:
    if not isinstance(item, dict):
        item = {"detect_phrase": str(fallback_phrase or item or "")}
    detect_phrase = str(item.get("detect_phrase") or fallback_phrase or item.get("object") or "").strip().lower()
    main_object = str(item.get("object") or detect_phrase).strip().lower()
    relation_context = str(item.get("relation_context") or "").strip().lower()
    reference_object = str(item.get("reference_object") or "").strip().lower()
    detect_phrase = re.sub(r"\s+", " ", detect_phrase).strip(" \"'`.,;:!?()[]{}")
    main_object = re.sub(r"\s+", " ", main_object).strip(" \"'`.,;:!?()[]{}")
    relation_context = re.sub(r"\s+", " ", relation_context).strip(" \"'`.,;:!?()[]{}")
    reference_object = re.sub(r"\s+", " ", reference_object).strip(" \"'`.,;:!?()[]{}")
    return {
        "detect_phrase": detect_phrase,
        "object": main_object or detect_phrase,
        "relation_context": relation_context,
        "reference_object": reference_object,
    }


def _parse_objects_json_block(response: str):
    match = re.search(r"\[Objects\]\s*", str(response or ""), flags=re.IGNORECASE)
    if not match:
        return None
    start = str(response).find("[", match.end())
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(str(response)[start:])
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None


def parse_vlm_object_extraction_response(response: str, preserve_duplicates: bool = False) -> tuple:
    detect_objects = parse_vlm_object_names(response, preserve_duplicates=preserve_duplicates)
    raw_items = _parse_objects_json_block(response)
    if raw_items is None:
        return detect_objects, []

    items = []
    for idx, item in enumerate(raw_items):
        fallback = detect_objects[idx] if idx < len(detect_objects) else ""
        normalized = _normalize_object_extraction_item(item, fallback_phrase=fallback)
        if normalized["detect_phrase"] and not _looks_like_non_object_phrase(normalized["detect_phrase"]):
            items.append(normalized)
    seen = {item["detect_phrase"] for item in items}
    for ref_item in list(items):
        reference = ref_item.get("reference_object", "")
        if reference and not _is_camera_reference(reference) and reference not in seen:
            items.append({"detect_phrase": reference, "object": reference, "relation_context": "", "reference_object": ""})
            seen.add(reference)
    if not items:
        return detect_objects, []
    return [item["detect_phrase"] for item in items], items


def extract_object_names_from_omni3d_question(question: str) -> list:
    names = []
    options_match = re.search(r"Options:\s*\{([^}]+)\}", question, flags=re.IGNORECASE)
    if options_match:
        option_names = [part.strip().lower() for part in options_match.group(1).split(",")]
        if not all(_is_bare_attribute_phrase(name) for name in option_names):
            names.extend(option_names)

    question_without_options = re.sub(r"Options:\s*\{[^}]+\}", "", question, flags=re.IGNORECASE)
    article_pattern = re.compile(
        r"\b(?:the|a|an)\s+(.+?)(?=\s+(?:is|are|was|were|would|will|could|should|to|of|on|in|under|above|below|behind|before|after|from|with|towards|than|or|and|as|directly|right|left|front|back)\b|[?,.]|$)",
        flags=re.IGNORECASE,
    )
    for match in article_pattern.findall(question_without_options):
        cleaned = _clean_omni3d_object_name(match)
        if cleaned and not _looks_like_non_object_phrase(cleaned):
            names.append(cleaned)

    names.extend(_extract_relation_operand_phrases(question_without_options))
    names.extend(_extract_relation_reference_object_names(question_without_options))
    names.extend(_extract_distribution_count_object_names(question_without_options))
    return _dedupe_preserve_order(names)


def _clean_omni3d_object_name(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("left-most", "leftmost").replace("top-most", "topmost")
    text = re.sub(r"\s+", " ", text)
    text = text.split(":", 1)[0]
    text = re.split(
        r"\b(?:would|will|could|should|do|does|did|have to|still|large enough|directly|without|first)\b",
        text,
        maxsplit=1,
    )[0]
    text = re.sub(r"^(combined|same|double)\s+", "", text)
    text = re.sub(r"\s+(?:in meters|in centimeters|in kilometers|on top of each other).*$", "", text)
    text = re.sub(r"\b(?:closer|farther|further|nearer|nearest|closest|furthest|farthest|next)$", "", text).strip()
    return text.strip(" \"'`({[])}.,;:!?")


def _looks_like_non_object_phrase(text: str) -> bool:
    text = str(text).strip().lower()
    if not text:
        return True
    measurement_heads = {
        "ratio",
        "height",
        "width",
        "length",
        "radius",
        "volume",
        "distance",
        "color",
        "number",
        "objects",
        "object",
        "one",
        "physical",
    }
    non_objects = {
        "left",
        "right",
        "front",
        "back",
        "behind",
        "above",
        "below",
        "up",
        "down",
        "yes",
        "no",
        "same",
        "same object types",
        "same object type",
        "object type",
        "object types",
    }
    first_word = text.split()[0]
    has_relation_context = bool(RELATION_CONNECTOR_RE.search(text))
    return (
        text in non_objects
        or first_word in measurement_heads
        or (not has_relation_context and (" of the " in text or " of a " in text))
        or (not has_relation_context and len(text.split()) > 6)
    )


def _dedupe_preserve_order(names: list) -> list:
    seen = set()
    output = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            output.append(name)
    return output


def write_sample_json(sample_dir: Path, sample_key: str, record: dict) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / "object_3d_positions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({sample_key: record}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output_path


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    if not samples:
        raise ValueError("No samples found in --dataset_json, --sample_json, or --jsonl")

    config = Object3DExtractionConfig()
    config.detection.box_threshold = args.box_threshold
    config.detection.text_threshold = args.text_threshold
    config.detection.use_vlm_refinement = args.use_vlm_refinement
    config.detection.count_max_instances = args.count_max_instances
    config.mask_fallback_mode = args.mask_fallback
    vlm_model = build_vlm_refinement_model(args)
    locator = Object3DLocator(config=config, device=args.device, vlm_model=vlm_model)
    output_root = Path(args.save_dir)
    written_files = []

    for idx, sample in enumerate(samples):
        sample_key = get_sample_key(sample, idx)
        sample_save_dir = output_root / sample_key
        image = None
        object_names = []
        try:
            image = resolve_image_path(args, sample)
            object_info = resolve_object_names(
                sample,
                image=image,
                vlm_model=vlm_model if args.use_vlm_object_extraction else None,
                use_vlm_object_extraction=args.use_vlm_object_extraction,
            )
            object_names = object_info["objects"]
            print(
                "[{}/{}] {} objects: {}".format(
                    idx + 1,
                    len(samples),
                    sample_key,
                    json.dumps(object_names, ensure_ascii=False),
                )
            )
            result = locator.extract(
                image=image,
                object_names=object_names,
                visualize=not args.no_visualize,
                save_dir=sample_save_dir,
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                question_type=object_info["question_type"],
                object_extraction_items=object_info.get("object_extraction_items"),
            )
            record = {
                "sample_id": sample_key,
                "image": image,
                "objects": object_names,
                "object_extraction_method": object_info["method"],
                "vlm_extracted_objects": object_info["vlm_extracted_objects"],
                "rule_extracted_objects": object_info["rule_extracted_objects"],
                "object_extraction_response": object_info["object_extraction_response"],
                "object_extraction_items": object_info.get("object_extraction_items", []),
                "question_type": object_info["question_type"],
                "result": result,
            }
        except Exception as exc:
            record = {
                "sample_id": sample_key,
                "image": image,
                "objects": object_names,
                "error": str(exc),
            }

        output_path = write_sample_json(sample_save_dir, sample_key, record)
        written_files.append(str(output_path))
        print("Wrote:", output_path)

    print(json.dumps({"written_files": written_files}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
