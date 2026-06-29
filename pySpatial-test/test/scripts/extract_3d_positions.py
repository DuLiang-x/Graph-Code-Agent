#!/usr/bin/env python
"""Extract Omni3D object 3D positions with VLM candidate scoring.

This entry point is self-contained. Ordinary non-counting targets use a
candidate-scoring VLM prompt instead of the standard refinement validator.
Counting targets still use Object3DLocator's existing multi-instance path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

REPO_TEST_DIR = Path(__file__).resolve().parents[1]
if str(REPO_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_TEST_DIR))

from object_3d_extraction import Object3DExtractionConfig, Object3DLocator
from object_3d_extraction.object_3d_locator import (
    clamp_box_to_image,
    draw_labeled_box,
    is_count_ratio_question,
    rank_candidates_for_object,
)
from object_3d_extraction.prompts import (
    PATTERN_GET_OBJECTS_OF_INTEREST,
    PROMPT_GET_OBJECTS_OF_INTEREST,
    PROMPT_GET_OBJECTS_OF_INTEREST_AUX,
    QUESTION_TYPE_RULES,
)
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
        )
        record = {
            "sample_id": sample_key,
            "image": resolved_image,
            "objects": object_names,
            "object_extraction_method": object_info["method"],
            "vlm_extracted_objects": object_info["vlm_extracted_objects"],
            "rule_extracted_objects": object_info["rule_extracted_objects"],
            "object_extraction_response": object_info["object_extraction_response"],
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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=768,
        help="Maximum tokens for the VLM candidate-scoring response.",
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
    r"\b(to the left of|to the right of|in front of|on top of|next to|right of|left of|beside|under|below|above|over|behind|inside|near)\b",
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
            vlm_objects, vlm_response = extract_object_names_with_vlm(image, question, vlm_model, question_type=question_type)
            vlm_objects = postprocess_extracted_objects(vlm_objects, question, question_type)
        except Exception as exc:
            vlm_response = "ERROR: {}".format(exc)

    if vlm_objects:
        return {
            "objects": vlm_objects,
            "method": "vlm",
            "vlm_extracted_objects": vlm_objects,
            "rule_extracted_objects": rule_objects,
            "object_extraction_response": vlm_response,
            "question_type": question_type,
        }

    if rule_objects:
        return {
            "objects": rule_objects,
            "method": "rule_fallback" if use_vlm_object_extraction else "rule",
            "vlm_extracted_objects": vlm_objects,
            "rule_extracted_objects": rule_objects,
            "object_extraction_response": vlm_response,
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
        objects = parse_vlm_object_names(response, preserve_duplicates=is_same_type_existence_question(question))
        if objects:
            return objects, response
    return [], response


def parse_vlm_object_names(response: str, preserve_duplicates: bool = False) -> list:
    matches = re.findall(PATTERN_GET_OBJECTS_OF_INTEREST, str(response or ""))
    if not matches:
        return []
    match = matches[-1]
    names = []
    for part in match.strip().replace("[", "").replace("]", "").split(","):
        cleaned = part.strip().lower().replace("'", "").replace('\"', "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and not _looks_like_non_object_phrase(cleaned):
            names.append(cleaned)
    return names if preserve_duplicates else _dedupe_preserve_order(names)


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


def build_scoring_config(args: argparse.Namespace) -> Object3DExtractionConfig:
    config = Object3DExtractionConfig()
    config.detection.box_threshold = args.box_threshold
    config.detection.text_threshold = args.text_threshold
    config.detection.use_vlm_refinement = args.use_vlm_refinement
    config.detection.count_max_instances = args.count_max_instances
    config.mask_fallback_mode = args.mask_fallback
    return config


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("_") or "object"


def clamp_box(box: List[int], image_size: Tuple[int, int]) -> List[int]:
    return [int(v) for v in clamp_box_to_image([int(x) for x in box], image_size)]


def make_candidate_overlay(image: Image.Image, candidates: List[Dict[str, Any]], selected_index: Optional[int] = None) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "blue"]
    for order, candidate in enumerate(candidates):
        box = clamp_box(candidate["box2d"], image.size)
        candidate_index = int(candidate.get("candidate_index", order))
        color = "lime" if selected_index is not None and candidate_index == selected_index else colors[order % len(colors)]
        width = 6 if selected_index is not None and candidate_index == selected_index else 4
        draw_labeled_box(draw, box, color, width=width)
        x1, y1 = box[0], box[1]
        draw.rectangle([x1, y1, x1 + 42, y1 + 28], fill="black")
        draw.text((x1 + 8, y1 + 5), str(candidate_index), fill=color)
    return overlay


def make_crop_grid(image: Image.Image, candidates: List[Dict[str, Any]]) -> Image.Image:
    crops = []
    for order, candidate in enumerate(candidates):
        box = clamp_box(candidate["box2d"], image.size)
        crop = image.crop(tuple(box)).resize((224, 224))
        draw = ImageDraw.Draw(crop)
        candidate_index = int(candidate.get("candidate_index", order))
        draw.rectangle([0, 0, 68, 30], fill="white")
        draw.text((8, 6), str(candidate_index), fill="red")
        crops.append(crop)
    cols = min(5, max(1, len(crops)))
    rows = (len(crops) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 224, rows * 224), "white")
    for idx, crop in enumerate(crops):
        grid.paste(crop, ((idx % cols) * 224, (idx // cols) * 224))
    return grid


def build_scoring_prompt(question: str, object_name: str, candidates: List[Dict[str, Any]]) -> str:
    rows = []
    for order, candidate in enumerate(candidates):
        candidate_index = int(candidate.get("candidate_index", order))
        prompt = candidate.get("prompt") or candidate.get("phrase") or ""
        rows.append(f'- index {candidate_index}: box2d={candidate.get("box2d")}, crop label={candidate_index}, detector prompt="{prompt}"')
    return f"""
You are scoring bounding-box candidates for Omni3D-Bench 3D object extraction.

Question context: {question}
Target object: {object_name}

You are given two images: first, the full image with numbered candidate boxes; second, a crop grid with the same candidate indexes.
Use only the images, box coordinates, target object name, and question context. Do not rely on detector scores or prior rank scores.

Candidate boxes:
{chr(10).join(rows)}

Scoring rules:
- Score each candidate from 0 to 100 for how well it matches the target object in the question.
- Consider exact object category, color, material, shape, side/position words, relation/reference phrases, and scene role.
- Relation phrases such as "under the TV", "next to the sofa", "left of the chair", or "rightmost" are selection constraints; they do not mean choosing the reference object itself.
- Prefer the complete visible target object when the target is a complete object.
- If the target object phrase is plural or group-like, such as "topmost cabinets", "cabinets to the left of the fume vent", "both sinks", or "two bedside tables", score the candidate higher only when it represents the requested group/multiple-instance operand, not just one member of that group.
- For plural/group targets, do not downgrade the target to a singular object. A single cabinet is not the best match for "topmost cabinets" when there is a candidate covering the top row/group of cabinets.
- Do not choose a box that contains multiple unrelated objects or large background regions unless the target is an area object.
- If the target object is naturally truncated by the camera viewpoint, occlusion, or image boundary, a partially visible but semantically correct box can be valid.
- Similar names are not interchangeable: distinguish TV vs TV stand, table vs tabletop, chair vs cushion, cabinet vs cabinet top, sofa vs pillow, and same-category objects with different colors or positions.

Return only valid JSON in this exact shape:
{{"scores":[{{"index":0,"score":85,"reason":"short reason"}}],"selected_index":0}}
Use INVALID as selected_index only if none of the candidates match the target.
""".strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:pos + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_scoring_response(response: str, valid_indices: List[int]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    data = extract_json_object(response)
    if not isinstance(data, dict):
        return None, None
    selected = data.get("selected_index")
    if isinstance(selected, str) and selected.strip().upper() == "INVALID":
        return None, data
    try:
        selected_int = int(selected)
    except (TypeError, ValueError):
        return None, data
    if selected_int not in valid_indices:
        return None, data
    return selected_int, data


def _choose_candidate_by_scoring_vlm(
    vlm_model,
    image_pil: Image.Image,
    object_name: str,
    candidates: List[Dict[str, object]],
    question: str,
    save_dir: Optional[Union[str, Path]],
    max_new_tokens: int,
) -> Dict[str, object]:
    overlay = make_candidate_overlay(image_pil, candidates)
    grid = make_crop_grid(image_pil, candidates)
    prompt = build_scoring_prompt(question, object_name, candidates)

    candidate_overlay_path = None
    candidate_grid_path = None
    selected_overlay_path = None
    if save_dir is not None:
        object_dir = Path(save_dir) / "candidate_scoring"
        object_dir.mkdir(parents=True, exist_ok=True)
        safe = safe_name(object_name)
        candidate_overlay_path = object_dir / "{}_candidate_overlay.png".format(safe)
        candidate_grid_path = object_dir / "{}_crop_grid.png".format(safe)
        overlay.save(candidate_overlay_path)
        grid.save(candidate_grid_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": overlay},
                {"type": "image", "image": grid},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = vlm_model.process_messages(messages, max_new_tokens=max_new_tokens)
    valid_indices = [int(candidate.get("candidate_index", idx)) for idx, candidate in enumerate(candidates)]
    selected_index, parsed = parse_scoring_response(response, valid_indices)

    selected_candidate = None
    if selected_index is not None:
        selected_candidate = next(
            (
                candidate
                for idx, candidate in enumerate(candidates)
                if int(candidate.get("candidate_index", idx)) == selected_index
            ),
            None,
        )
        if selected_candidate is not None and save_dir is not None:
            selected_overlay = make_candidate_overlay(image_pil, candidates, selected_index=selected_index)
            selected_overlay_path = Path(save_dir) / "candidate_scoring" / "{}_vlm_selected_overlay.png".format(safe_name(object_name))
            selected_overlay.save(selected_overlay_path)

    return {
        "selected_index": selected_index,
        "selected_candidate": selected_candidate,
        "raw_response": response,
        "parsed_response": parsed,
        "candidate_overlay_path": str(candidate_overlay_path) if candidate_overlay_path else None,
        "candidate_grid_path": str(candidate_grid_path) if candidate_grid_path else None,
        "selected_overlay_path": str(selected_overlay_path) if selected_overlay_path else None,
    }


class ScoringObject3DLocator(Object3DLocator):
    def __init__(
        self,
        config: Optional[Union[Object3DExtractionConfig, Dict[str, Any]]] = None,
        device: str = "cuda",
        detection_module: Optional[Any] = None,
        depth_module: Optional[Any] = None,
        orientation_module: Optional[Any] = None,
        vlm_model: Optional[Any] = None,
        scoring_max_new_tokens: int = 768,
    ):
        super().__init__(
            config=config,
            device=device,
            detection_module=detection_module,
            depth_module=depth_module,
            orientation_module=orientation_module,
            vlm_model=vlm_model,
        )
        self.scoring_max_new_tokens = scoring_max_new_tokens

    def _select_detection(
        self,
        image_pil: Image.Image,
        object_name: str,
        candidates: List[Dict[str, object]],
        selected_boxes: Dict[str, List[int]],
        save_dir: Optional[Union[str, Path]],
        question: str,
        forced_candidate_selections: Optional[Dict[str, int]] = None,
    ) -> Dict[str, object]:
        rule_ranked = rank_candidates_for_object(image_pil, object_name, candidates, selected_boxes)
        rule_selected = dict(rule_ranked[0])
        rule_selected["_ranked_candidates"] = rule_ranked
        rule_selected["rule_selected_index"] = 0
        rule_selected["vlm_selected_index"] = None
        rule_selected["final_selected_index"] = 0
        rule_selected["selection_decision"] = "rule_ranker_no_scoring_model"
        rule_selected["selection_reject_reason"] = None
        rule_selected["rank_reason"] = "rule_ranker"

        if self.vlm_model is None or not self.config.detection.use_vlm_refinement:
            return rule_selected

        try:
            scoring = _choose_candidate_by_scoring_vlm(
                self.vlm_model,
                image_pil,
                object_name,
                rule_ranked,
                question,
                save_dir,
                self.scoring_max_new_tokens,
            )
        except Exception as exc:
            rule_selected["selection_decision"] = "rule_ranker_scoring_error"
            rule_selected["selection_reject_reason"] = str(exc)
            return rule_selected

        for key in ("candidate_overlay_path", "candidate_grid_path", "selected_overlay_path"):
            if scoring.get(key):
                rule_selected[key] = scoring[key]

        selected_candidate = scoring.get("selected_candidate")
        if selected_candidate is None:
            rule_selected["selection_decision"] = "rule_ranker_scoring_invalid"
            rule_selected["selection_reject_reason"] = "invalid_vlm_candidate_scoring_response"
            rule_selected["vlm_response"] = scoring.get("raw_response")
            rule_selected["vlm_candidate_scores"] = scoring.get("parsed_response")
            return rule_selected

        selected = dict(selected_candidate)
        selected["_ranked_candidates"] = rule_ranked
        selected["rule_selected_index"] = 0
        selected["vlm_selected_index"] = scoring["selected_index"]
        selected["final_selected_index"] = scoring["selected_index"]
        selected["selection_decision"] = "vlm_candidate_scoring"
        selected["selection_reject_reason"] = None
        selected["rank_reason"] = "vlm_candidate_scoring"
        selected["vlm_response"] = scoring.get("raw_response")
        selected["vlm_candidate_scores"] = scoring.get("parsed_response")
        for key in ("candidate_overlay_path", "candidate_grid_path", "selected_overlay_path"):
            if scoring.get(key):
                selected[key] = scoring[key]
        return selected


class ScoringObjectExtractionRunner:
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
        max_new_tokens: int = 768,
        box_threshold: float = 0.05,
        text_threshold: float = 0.05,
        image_root: str = "/data/datasets/Omni3D-Bench/images",
        base_data_path: str = None,
    ):
        print("Initializing scoring-based object extractor...")
        self.device = device
        self.mask_fallback = mask_fallback
        self.args = argparse.Namespace(
            image=None,
            image_root=image_root,
            base_data_path=base_data_path,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
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
        self.locator = ScoringObject3DLocator(
            config=config,
            device=device,
            vlm_model=vlm_model,
            scoring_max_new_tokens=max_new_tokens,
        )

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
        )
        record = {
            "sample_id": sample_key,
            "image": resolved_image,
            "objects": object_names,
            "object_extraction_method": "{}+vlm_candidate_scoring".format(object_info["method"]),
            "vlm_extracted_objects": object_info["vlm_extracted_objects"],
            "rule_extracted_objects": object_info["rule_extracted_objects"],
            "object_extraction_response": object_info["object_extraction_response"],
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
    count_max_instances: int = 20,
    max_new_tokens: int = 768,
):
    if extractor is not None:
        return extractor.extract_sample(
            sample,
            output_root=output_root,
            visualize=visualize,
            image=image,
            base_data_path=base_data_path,
        )

    runner = ScoringObjectExtractionRunner(
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
        count_max_instances=count_max_instances,
        max_new_tokens=max_new_tokens,
        base_data_path=base_data_path,
    )
    return runner.extract_sample(
        sample,
        output_root=output_root,
        visualize=visualize,
        image=image,
        base_data_path=base_data_path,
    )


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    if not samples:
        raise ValueError("No samples found in --dataset_json, --sample_json, or --jsonl")

    output_root = Path(args.save_dir)
    config = build_scoring_config(args)
    vlm_model = build_vlm_refinement_model(args)
    locator = ScoringObject3DLocator(
        config=config,
        device=args.device,
        vlm_model=vlm_model,
        scoring_max_new_tokens=args.max_new_tokens,
    )
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
            )
            record = {
                "sample_id": sample_key,
                "image": image,
                "objects": object_names,
                "object_extraction_method": "{}+vlm_candidate_scoring".format(object_info["method"]),
                "vlm_extracted_objects": object_info["vlm_extracted_objects"],
                "rule_extracted_objects": object_info["rule_extracted_objects"],
                "object_extraction_response": object_info["object_extraction_response"],
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
