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
            vlm_model_path=vlm_model_path,
            mask_fallback=mask_fallback,
        )
        config = Object3DExtractionConfig()
        config.detection.box_threshold = self.args.box_threshold
        config.detection.text_threshold = self.args.text_threshold
        config.detection.use_vlm_refinement = use_vlm_refinement
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
YES_NO_ANSWER_RE = re.compile(r"^\s*(yes|no)\s*[.!]?\s*$", re.IGNORECASE)


def _sample_answer(sample):
    for key in ("answer", "expected_answer", "gt_answer"):
        if key in sample and sample.get(key) is not None:
            return sample.get(key)
    return None


def _is_yes_no_answer(answer) -> bool:
    return bool(YES_NO_ANSWER_RE.match(str(answer or "")))


def classify_question_type(sample) -> str:
    question = sample.get("question", "")
    answer_type = str(sample.get("answer_type") or "").lower()
    answer = _sample_answer(sample)

    if _is_yes_no_answer(answer):
        return "yes_no"
    if answer_type == "float":
        return "numeric_other"
    if answer_type == "int" or COUNT_QUESTION_RE.search(question):
        return "numeric_ct"
    if answer_type == "str":
        return "choice_object"
    return "generic"


def build_object_extraction_prompt(question: str, question_type: str = "generic") -> str:
    rules = QUESTION_TYPE_RULES.get(question_type, QUESTION_TYPE_RULES["generic"])
    return PROMPT_GET_OBJECTS_OF_INTEREST.format(
        question=question,
        question_type_rules=rules.strip(),
    )


def resolve_object_names(
    sample,
    image: str = None,
    vlm_model=None,
    use_vlm_object_extraction: bool = True,
) -> dict:
    question = sample.get("question", "")
    question_type = classify_question_type(sample)
    rule_objects = resolve_object_names_by_rule(sample)
    vlm_objects = []
    vlm_response = None

    if use_vlm_object_extraction and vlm_model is not None and image is not None:
        try:
            vlm_objects, vlm_response = extract_object_names_with_vlm(image, question, vlm_model, question_type=question_type)
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
        objects = parse_vlm_object_names(response)
        if objects:
            return objects, response
    return [], response


def parse_vlm_object_names(response: str) -> list:
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
    return _dedupe_preserve_order(names)


def extract_object_names_from_omni3d_question(question: str) -> list:
    names = []
    options_match = re.search(r"Options:\s*\{([^}]+)\}", question, flags=re.IGNORECASE)
    if options_match:
        names.extend(part.strip().lower() for part in options_match.group(1).split(","))

    question_without_options = re.sub(r"Options:\s*\{[^}]+\}", "", question, flags=re.IGNORECASE)
    article_pattern = re.compile(
        r"\b(?:the|a|an)\s+(.+?)(?=\s+(?:is|are|was|were|would|will|could|should|to|of|on|in|under|above|below|behind|before|after|from|with|towards|than|or|and|as|directly|right|left|front|back)\b|[?,.]|$)",
        flags=re.IGNORECASE,
    )
    for match in article_pattern.findall(question_without_options):
        cleaned = _clean_omni3d_object_name(match)
        if cleaned and not _looks_like_non_object_phrase(cleaned):
            names.append(cleaned)

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
    }
    first_word = text.split()[0]
    return (
        text in non_objects
        or first_word in measurement_heads
        or " of the " in text
        or " of a " in text
        or len(text.split()) > 6
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
            )
            record = {
                "sample_id": sample_key,
                "image": image,
                "objects": object_names,
                "object_extraction_method": object_info["method"],
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
