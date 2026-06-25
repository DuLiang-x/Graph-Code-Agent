
#!/usr/bin/env python3
"""
Evaluate pySpatial Agent on MindCube dataset and calculate statistics for three types:
- among: from image paths like "other_all_image/among/shoe_216/front_007.jpg"
- around: from image paths like "other_all_image/around/26b1a4b226e2e3509100a595ebc5d17dafd361abfdf06fcf20e36f905e138faa/2_frame_00166.png"
- rotation: from image paths containing "rotation"
"""

import os
import sys
import json
import argparse
import math
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import threading
import textwrap
import tempfile
import traceback
try:
    import backoff
except ImportError:
    class _BackoffFallback:
        full_jitter = None

        @staticmethod
        def expo(*args, **kwargs):
            return None

        @staticmethod
        def on_exception(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    backoff = _BackoffFallback()

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pySpatial_Interface import Agent, Scene, pySpatial, DEFAULT_LOCAL_QWEN_MODEL_PATH, OBJECT_OUTPUT_ROOTS, LEGACY_OBJECT_OUTPUT_ROOTS
from scripts.demo_extract_3d_positions import classify_question_type

# Rate limiting globals
last_request_time = 0
min_request_interval = 0.1  # Minimum time between requests (100ms)
request_lock = threading.Lock()
CALL_AGENT_MAX_TRIES = 5


def _log_agent_retry(details):
    target = details.get("target")
    target_name = getattr(target, "__name__", "call_agent_with_retry")
    tries = details.get("tries", "?")
    print(f"Retrying {target_name} (attempt {tries}/{CALL_AGENT_MAX_TRIES})...")


def rate_limit():
    """Apply rate limiting between API requests"""
    global last_request_time
    
    with request_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < min_request_interval:
            time.sleep(min_request_interval - time_since_last)
        last_request_time = time.time()


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=CALL_AGENT_MAX_TRIES,
    factor=2,
    jitter=backoff.full_jitter,
    on_backoff=_log_agent_retry
)
def call_agent_with_retry(agent, method_name, *args, **kwargs):
    """Call agent method with rate limiting and retry logic"""
    rate_limit()
    
    try:
        method = getattr(agent, method_name)
        result = method(*args, **kwargs)
        return result
    except Exception as e:
        error_msg = str(e)
        
        # Handle rate limit specifically
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            print(f"Rate limit hit for {method_name}, waiting before retry...")
            time.sleep(60) 
        elif "tokens per min" in error_msg.lower():
            print(f"Token rate limit hit for {method_name}, waiting before retry...")
            time.sleep(60)  
            raise  
        else:
            # For other errors, re-raise without modification
            raise


def extract_type_from_images(images: List[str]) -> str:
    """
    Extract the type (among, around, rotation) from the image paths.
    
    Args:
        images: List of image paths
        
    Returns:
        The type string or 'unknown' if cannot be determined
    """
    for image_path in images:
        if 'among' in image_path:
            return 'among'
        elif 'around' in image_path:
            return 'around'
        elif 'rotation' in image_path:
            return 'rotation'
    
    return 'unknown'


def normalize_answer_text(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\.,;:!?]+|[\s\.,;:!?]+$", "", text)
    return text


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _extract_number(value):
    text = str(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match:
        return float(match.group(0))
    normalized = normalize_answer_text(text)
    for token in re.findall(r"[a-z]+", normalized):
        if token in _NUMBER_WORDS:
            return float(_NUMBER_WORDS[token])
    return None


def _extract_yes_no(value):
    normalized = normalize_answer_text(value)
    match = re.match(r"^(yes|no)\b", normalized)
    return match.group(1) if match else None


def evaluate_answer_correctness(generated_answer, expected_answer, answer_type: str = None, abs_tol: float = 0.05, rel_tol: float = 0.05) -> bool:
    if expected_answer is None or generated_answer is None:
        return False
    if answer_type == "float" or isinstance(expected_answer, float):
        expected = _extract_number(expected_answer)
        generated = _extract_number(generated_answer)
        if expected is None or generated is None:
            return False
        return math.isclose(generated, expected, abs_tol=abs_tol, rel_tol=rel_tol)

    expected_text = normalize_answer_text(expected_answer)
    if expected_text in {"yes", "no"}:
        generated_yes_no = _extract_yes_no(generated_answer)
        return generated_yes_no == expected_text

    return normalize_answer_text(generated_answer) == expected_text


NUMERIC_ANSWER_TYPES = {"float", "int", "number", "numeric"}


def is_numeric_answer_type(answer_type: str = None, expected_answer=None) -> bool:
    normalized_type = str(answer_type or "").strip().lower()
    if normalized_type in NUMERIC_ANSWER_TYPES:
        return True
    if isinstance(expected_answer, (int, float)) and not isinstance(expected_answer, bool):
        return True
    return _extract_number(expected_answer) is not None and normalized_type not in {"str", "string", "yes_no", "choice"}


def compute_numeric_relative_error(generated_answer, expected_answer) -> Optional[float]:
    expected = _extract_number(expected_answer)
    generated = _extract_number(generated_answer)
    if expected is None or generated is None or expected == 0:
        return None
    return abs(generated - expected) / abs(expected)


def compute_numeric_mra(generated_answer, expected_answer, thresholds: Optional[List[float]] = None) -> Optional[float]:
    relative_error = compute_numeric_relative_error(generated_answer, expected_answer)
    if relative_error is None:
        return None
    thresholds = thresholds or [0.5 + 0.05 * idx for idx in range(10)]
    passed = sum(1 for theta in thresholds if relative_error < 1 - theta)
    return passed / float(len(thresholds)) if thresholds else None


def compute_numeric_metrics(generated_answer, expected_answer, answer_type: str = None) -> Dict[str, Optional[float]]:
    if not is_numeric_answer_type(answer_type, expected_answer):
        return {
            "numeric_relative_error": None,
            "numeric_mra": None,
            "numeric_score": None,
            "float_relative_error": None,
            "float_mra": None,
        }
    relative_error = compute_numeric_relative_error(generated_answer, expected_answer)
    mra = compute_numeric_mra(generated_answer, expected_answer)
    is_float = str(answer_type or "").strip().lower() == "float" or isinstance(expected_answer, float)
    return {
        "numeric_relative_error": relative_error,
        "numeric_mra": mra,
        "numeric_score": mra,
        "float_relative_error": relative_error if is_float else None,
        "float_mra": mra if is_float else None,
    }


def compute_float_relative_error(generated_answer, expected_answer) -> Optional[float]:
    return compute_numeric_relative_error(generated_answer, expected_answer)


def compute_float_mra(generated_answer, expected_answer, thresholds: Optional[List[float]] = None) -> Optional[float]:
    return compute_numeric_mra(generated_answer, expected_answer, thresholds=thresholds)


def compute_float_metrics(generated_answer, expected_answer, answer_type: str = None) -> Dict[str, Optional[float]]:
    metrics = compute_numeric_metrics(generated_answer, expected_answer, answer_type)
    return {
        "float_relative_error": metrics["float_relative_error"],
        "float_mra": metrics["float_mra"],
    }


def compute_acc_mra(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = 0
    score = 0.0
    for result in results:
        if result.get("expected_answer") is None or result.get("generated_answer") is None:
            continue
        if is_numeric_answer_type(result.get("answer_type"), result.get("expected_answer")):
            mra = result.get("numeric_mra")
            if mra is None:
                mra = result.get("float_mra")
            if mra is None:
                continue
            score += float(mra)
            total += 1
        else:
            score += 1.0 if result.get("answer_correct") else 0.0
            total += 1
    value = round(score / total, 6) if total else 0.0
    return {"acc_mra": value, "acc_mra_count": total, "mra_score": value, "mra_score_count": total}


QUESTION_TYPE_ORDER = ["numeric_ct", "numeric_other", "yes_no", "choice_object"]
QUESTION_TYPE_ALIASES = {"count_ratio": "numeric_ct"}


def normalize_question_type_for_summary(question_type: str = None) -> str:
    value = str(question_type or "").strip().lower()
    value = QUESTION_TYPE_ALIASES.get(value, value)
    if value in set(QUESTION_TYPE_ORDER):
        return value
    return "unknown"


def infer_question_type(question: str, expected_answer=None, answer_type: str = None, object_3d_boxes: Dict[str, Any] = None) -> str:
    if isinstance(object_3d_boxes, dict):
        for payload in object_3d_boxes.values():
            if isinstance(payload, dict):
                if payload.get("question_type"):
                    return normalize_question_type_for_summary(payload.get("question_type"))
                if isinstance(payload.get("result"), dict) and payload.get("question_type"):
                    return normalize_question_type_for_summary(payload.get("question_type"))
    sample = {"question": question, "answer": expected_answer, "answer_type": answer_type}
    return normalize_question_type_for_summary(classify_question_type(sample))


def _load_flowchart_font(size: int, mono: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" if mono else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf" if mono else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _wrap_flowchart_text(text: str, width: int = 110, max_lines: int = 180) -> List[str]:
    lines = []
    for raw_line in str(text or "").splitlines() or [""]:
        wrapped = textwrap.wrap(raw_line, width=width, replace_whitespace=False, drop_whitespace=False)
        lines.extend(wrapped or [""])
        if len(lines) >= max_lines:
            return lines[:max_lines] + ["... truncated ..."]
    return lines


def _blank_debug_panel(result: Dict[str, Any], width: int = 1400, height: int = 900):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_flowchart_font(28)
    body_font = _load_flowchart_font(22)
    draw.text((30, 30), "3D Debug Panel", fill=(20, 20, 20), font=title_font)
    draw.text((30, 85), "No object_3d_boxes were available for this sample.", fill=(60, 60, 60), font=body_font)
    y = 140
    for line in _wrap_flowchart_text("Question: " + str(result.get("question", "")), width=90, max_lines=16):
        draw.text((30, y), line, fill=(30, 30, 30), font=body_font)
        y += 30
    return image


def _append_flowchart_footer(panel_path: str, output_path: Path, result: Dict[str, Any]) -> str:
    from PIL import Image, ImageDraw

    base = Image.open(panel_path).convert("RGB")
    title_font = _load_flowchart_font(24)
    mono_font = _load_flowchart_font(18, mono=True)
    body_font = _load_flowchart_font(20)
    padding = 30
    line_height = 24

    code = result.get("generated_code") or "N/A"
    answer = result.get("generated_answer") or "N/A"
    reasoning = result.get("answer_reasoning") or "N/A"
    correctness = "Correct" if result.get("answer_correct") else "Incorrect"
    fallback = "yes" if result.get("fallback_used") else "no"
    error = result.get("error") or "N/A"
    repair = "yes" if result.get("code_repair_used") else "no"
    repair_attempts = result.get("code_repair_attempts", 0)

    sections = [
        ("CodeAgent Generated Code", _wrap_flowchart_text(code, width=120, max_lines=120), mono_font),
        ("Generated Answer", _wrap_flowchart_text(answer, width=120, max_lines=20), body_font),
        ("Answer Reasoning", _wrap_flowchart_text(reasoning, width=120, max_lines=35), body_font),
        ("Evaluation", _wrap_flowchart_text(f"Expected: {result.get('expected_answer')} | Correctness: {correctness} | Fallback used: {fallback} | Code repair: {repair} ({repair_attempts}) | Error: {error}", width=120, max_lines=12), body_font),
    ]

    footer_height = padding
    for _, lines, _ in sections:
        footer_height += 34 + max(1, len(lines)) * line_height + 18
    footer_height += padding

    canvas = Image.new("RGB", (base.width, base.height + footer_height), "white")
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = base.height + padding
    draw.rectangle((0, base.height, base.width, base.height + footer_height), fill=(248, 248, 248))

    for title, lines, font in sections:
        draw.text((padding, y), title, fill=(20, 20, 20), font=title_font)
        y += 34
        for line in lines:
            draw.text((padding, y), line, fill=(35, 35, 35), font=font)
            y += line_height
        y += 18

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def _sample_extract_debug_dir(scene_id: str, mask_fallback: str = "auto", extract_output_dir: str = None) -> Optional[Path]:
    if not scene_id:
        return None
    candidates = []
    if extract_output_dir:
        candidates.append(Path(extract_output_dir) / scene_id)
    root = OBJECT_OUTPUT_ROOTS.get(mask_fallback)
    if root is not None:
        candidates.append(Path(root) / scene_id)
    legacy_root = LEGACY_OBJECT_OUTPUT_ROOTS.get(mask_fallback)
    if legacy_root is not None:
        candidates.append(Path(legacy_root) / scene_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def create_sample_flowchart(result: Dict[str, Any], flowcharts_dir: Path, extract_output_dir: str = None) -> Optional[str]:
    scene_id = result.get("scene_id", "unknown")
    output_path = flowcharts_dir / f"{scene_id}_flowchart.png"
    object_boxes = result.get("object_3d_boxes") or {}
    images = result.get("images") or []
    image_path = next((image for image in images if image and os.path.exists(image)), None)
    sample_debug_dir = _sample_extract_debug_dir(scene_id, result.get("mask_fallback", "auto"), extract_output_dir or result.get("extract_output_dir"))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if object_boxes and image_path:
                from object_3d_extraction.visualize_3d_aabb import compose_question_image_3d_visualization, render_3d_aabb_scene

                tmp_path = Path(tmpdir)
                existing_aabb_path = sample_debug_dir / "3d_aabb.png" if sample_debug_dir else None
                if existing_aabb_path and existing_aabb_path.exists():
                    aabb_path = existing_aabb_path
                else:
                    aabb_path = tmp_path / "3d_aabb.png"
                    render_3d_aabb_scene(object_boxes, str(aabb_path))

                panel_path = tmp_path / "3d_debug_panel.png"
                answer_text = "GT: {}\nGenerated: {}".format(result.get("expected_answer"), result.get("generated_answer"))
                compose_question_image_3d_visualization(
                    image_path,
                    result.get("question", ""),
                    str(aabb_path),
                    str(panel_path),
                    answer=answer_text,
                    results=object_boxes,
                    debug_dir=str(sample_debug_dir) if sample_debug_dir and sample_debug_dir.exists() else None,
                )
            else:
                blank = _blank_debug_panel(result)
                panel_path = os.path.join(tmpdir, "3d_debug_panel.png")
                blank.save(panel_path)
            return _append_flowchart_footer(str(panel_path), output_path, result)
    except Exception as exc:
        print(f"[{scene_id}] Failed to create flowchart: {exc}")
        return None


SUMMARY_RESULT_KEYS = [
    "scene_id",
    "question",
    "images",
    "expected_answer",
    "answer_type",
    "question_type",
    "generated_answer",
    "answer_reasoning",
    "answer_source",
    "answer_correct",
    "float_relative_error",
    "float_mra",
    "numeric_relative_error",
    "numeric_mra",
    "numeric_score",
    "parse_success",
    "execution_success",
    "answer_generation_success",
    "fallback_used",
    "generated_code",
    "flowchart_path",
    "error",
    "mode",
    "mask_fallback",
    "extract_output_dir",
    "force_extract",
    "use_vlm_object_extraction",
    "code_repair_used",
    "code_repair_attempts",
    "code_repair_error",
    "initial_generated_code",
    "initial_generated_response",
]


def summarize_result_for_output(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: result.get(key) for key in SUMMARY_RESULT_KEYS if key in result}


def compute_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_stats = defaultdict(lambda: {
        'total': 0.0,
        'parse_success': 0.0,
        'execution_success': 0.0,
        'answer_generation_success': 0.0,
        'correct_answers': 0.0,
        'evaluable_answers': 0.0,
        'errors': 0.0
    })
    question_type_stats = defaultdict(lambda: {
        'total': 0.0,
        'parse_success': 0.0,
        'execution_success': 0.0,
        'answer_generation_success': 0.0,
        'correct_answers': 0.0,
        'evaluable_answers': 0.0,
        'errors': 0.0,
        'mra_sum': 0.0,
        'mra_count': 0.0,
    })
    overall_stats = {
        'total_processed': 0,
        'parse_success': 0,
        'execution_success': 0,
        'answer_generation_success': 0,
        'correct_answers': 0,
        'evaluable_answers': 0,
        'errors': 0
    }

    for result in results:
        scene_type = result.get('scene_type', 'unknown')
        type_stats[scene_type]['total'] += 1
        overall_stats['total_processed'] += 1
        if result.get('error'):
            type_stats[scene_type]['errors'] += 1
            overall_stats['errors'] += 1
        if result.get('parse_success'):
            type_stats[scene_type]['parse_success'] += 1
            overall_stats['parse_success'] += 1
        if result.get('execution_success'):
            type_stats[scene_type]['execution_success'] += 1
            overall_stats['execution_success'] += 1
        if result.get('answer_generation_success'):
            type_stats[scene_type]['answer_generation_success'] += 1
            overall_stats['answer_generation_success'] += 1

        question_type = normalize_question_type_for_summary(result.get('question_type'))
        question_type_stats[question_type]['total'] += 1
        if result.get('error'):
            question_type_stats[question_type]['errors'] += 1
        if result.get('parse_success'):
            question_type_stats[question_type]['parse_success'] += 1
        if result.get('execution_success'):
            question_type_stats[question_type]['execution_success'] += 1
        if result.get('answer_generation_success'):
            question_type_stats[question_type]['answer_generation_success'] += 1

        if result.get('expected_answer') is not None and result.get('generated_answer') is not None:
            type_stats[scene_type]['evaluable_answers'] += 1
            overall_stats['evaluable_answers'] += 1
            question_type_stats[question_type]['evaluable_answers'] += 1
            if result.get('answer_correct'):
                type_stats[scene_type]['correct_answers'] += 1
                overall_stats['correct_answers'] += 1
                question_type_stats[question_type]['correct_answers'] += 1
            if question_type in {'numeric_ct', 'numeric_other'} and result.get('numeric_mra') is not None:
                question_type_stats[question_type]['mra_sum'] += float(result.get('numeric_mra'))
                question_type_stats[question_type]['mra_count'] += 1

    type_metrics = {}
    for scene_type, stats in type_stats.items():
        total = stats['total']
        type_metrics[scene_type] = {
            'count': total,
            'parse_rate': round(stats['parse_success'] / total * 100, 2) if total > 0 else 0,
            'execution_rate': round(stats['execution_success'] / total * 100, 2) if total > 0 else 0,
            'answer_generation_rate': round(stats['answer_generation_success'] / total * 100, 2) if total > 0 else 0,
            'correctness_rate': round(stats['correct_answers'] / stats['evaluable_answers'] * 100, 2) if stats['evaluable_answers'] > 0 else 0,
            'error_rate': round(stats['errors'] / total * 100, 2) if total > 0 else 0,
            'evaluable_count': stats['evaluable_answers'],
            'error_count': stats['errors']
        }

    question_type_metrics = {}
    for qtype in QUESTION_TYPE_ORDER + sorted(set(question_type_stats.keys()) - set(QUESTION_TYPE_ORDER)):
        stats = question_type_stats.get(qtype)
        if not stats:
            continue
        total_q = stats['total']
        metrics = {
            'count': total_q,
            'evaluable_count': stats['evaluable_answers'],
            'correct_answers': stats['correct_answers'],
            'correctness_rate': round(stats['correct_answers'] / stats['evaluable_answers'] * 100, 2) if stats['evaluable_answers'] > 0 else 0,
            'parse_rate': round(stats['parse_success'] / total_q * 100, 2) if total_q > 0 else 0,
            'execution_rate': round(stats['execution_success'] / total_q * 100, 2) if total_q > 0 else 0,
            'answer_generation_rate': round(stats['answer_generation_success'] / total_q * 100, 2) if total_q > 0 else 0,
            'error_rate': round(stats['errors'] / total_q * 100, 2) if total_q > 0 else 0,
            'error_count': stats['errors'],
        }
        if qtype in {'numeric_ct', 'numeric_other'}:
            metrics['mra'] = round(stats['mra_sum'] / stats['mra_count'], 6) if stats['mra_count'] > 0 else 0.0
            metrics['mra_count'] = stats['mra_count']
        question_type_metrics[qtype] = metrics

    acc_mra_metrics = compute_acc_mra(results)
    total = overall_stats['total_processed']
    overall_metrics = {
        'total_count': total,
        'parse_rate': round(overall_stats['parse_success'] / total * 100, 2) if total > 0 else 0,
        'execution_rate': round(overall_stats['execution_success'] / total * 100, 2) if total > 0 else 0,
        'answer_generation_rate': round(overall_stats['answer_generation_success'] / total * 100, 2) if total > 0 else 0,
        'correctness_rate': round(overall_stats['correct_answers'] / overall_stats['evaluable_answers'] * 100, 2) if overall_stats['evaluable_answers'] > 0 else 0,
        'acc_mra': acc_mra_metrics['acc_mra'],
        'acc_mra_count': acc_mra_metrics['acc_mra_count'],
        'mra_score': acc_mra_metrics['mra_score'],
        'mra_score_count': acc_mra_metrics['mra_score_count'],
        'error_rate': round(overall_stats['errors'] / total * 100, 2) if total > 0 else 0,
        'evaluable_count': overall_stats['evaluable_answers'],
        'error_count': overall_stats['errors']
    }
    return {
        'type_stats': type_stats,
        'type_metrics': type_metrics,
        'question_type_metrics': question_type_metrics,
        'overall_stats': overall_stats,
        'overall_metrics': overall_metrics,
    }


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    if hasattr(value, "item"):
        return make_json_safe(value.item())
    return value


def omni3d_entry_to_pipeline_entry(sample: Dict[str, Any], image_root: str) -> Dict[str, Any]:
    question_index = sample.get("question_index", sample.get("id", "unknown"))
    image_filename = sample.get("image_filename")
    images = sample.get("images") or []
    if image_filename:
        image_path = Path(image_filename)
        if not image_path.is_absolute():
            image_path = Path(image_root) / image_path
        images = [str(image_path)]
    return {
        "id": "omni3d_{}".format(question_index),
        "question": sample.get("question", ""),
        "images": images,
        "answer": sample.get("answer"),
        "answer_type": sample.get("answer_type"),
        "raw_sample": sample,
    }


def load_omni3d_entries(dataset_json: str, image_root: str, max_entries: int = None) -> List[Dict[str, Any]]:
    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("questions", data if isinstance(data, list) else [])
    entries = [omni3d_entry_to_pipeline_entry(sample, image_root) for sample in samples]
    return entries[:max_entries] if max_entries is not None else entries


def get_entry_question_index(entry: Dict[str, Any]) -> Optional[int]:
    raw_sample = entry.get("raw_sample") if isinstance(entry, dict) else None
    if isinstance(raw_sample, dict) and raw_sample.get("question_index") is not None:
        try:
            return int(raw_sample.get("question_index"))
        except (TypeError, ValueError):
            return None

    if entry.get("question_index") is not None:
        try:
            return int(entry.get("question_index"))
        except (TypeError, ValueError):
            return None

    entry_id = str(entry.get("id", ""))
    match = re.match(r"^omni3d_(\d+)$", entry_id)
    if match:
        return int(match.group(1))
    return None


def validate_index_range(start_index: Optional[int], end_index: Optional[int]) -> None:
    if start_index is not None and start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if end_index is not None and end_index < 0:
        raise ValueError("--end_index must be non-negative")
    if start_index is not None and end_index is not None and start_index > end_index:
        raise ValueError("--start_index must be less than or equal to --end_index")


def filter_entries_by_index_range(
    entries: List[Dict[str, Any]],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if start_index is None and end_index is None:
        return entries

    filtered = []
    for entry in entries:
        question_index = get_entry_question_index(entry)
        if question_index is None:
            continue
        if start_index is not None and question_index < start_index:
            continue
        if end_index is not None and question_index > end_index:
            continue
        filtered.append(entry)
    return filtered


def apply_index_range_and_max_entries(
    entries: List[Dict[str, Any]],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    max_entries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    entries = filter_entries_by_index_range(entries, start_index=start_index, end_index=end_index)
    return entries[:max_entries] if max_entries is not None else entries


_WORKER_OBJECT_EXTRACTORS = {}


class LazyObjectExtractor:
    def __init__(
        self,
        device: str,
        mask_fallback: str,
        model_path: str,
        backend: str,
        api_model: str,
        api_key: str = None,
        base_url: str = None,
        use_vlm_object_extraction: bool = True,
    ):
        self.kwargs = {
            "device": device,
            "mask_fallback": mask_fallback,
            "use_vlm_refinement": True,
            "vlm_model_path": model_path,
            "backend": backend,
            "api_model": api_model,
            "api_key": api_key,
            "base_url": base_url,
            "use_vlm_object_extraction": use_vlm_object_extraction,
        }
        self._runner = None

    def _get_runner(self):
        if self._runner is None:
            from scripts.demo_extract_3d_positions import ObjectExtractionRunner

            self._runner = ObjectExtractionRunner(**self.kwargs)
        return self._runner

    def extract_sample(self, *args, **kwargs):
        return self._get_runner().extract_sample(*args, **kwargs)


def build_object_extractor(
    mode: str,
    device: str,
    mask_fallback: str,
    model_path: str,
    backend: str,
    api_model: str,
    api_key: str = None,
    base_url: str = None,
    use_vlm_object_extraction: bool = True,
):
    if mode != "graph":
        return None
    return LazyObjectExtractor(
        device=device,
        mask_fallback=mask_fallback,
        model_path=model_path,
        backend=backend,
        api_model=api_model,
        api_key=api_key,
        base_url=base_url,
        use_vlm_object_extraction=use_vlm_object_extraction,
    )


def get_worker_object_extractor(config: Dict[str, Any]):
    if config.get("mode", "reconstruct") != "graph":
        return None
    key = (
        config.get("device", "cuda"),
        config.get("mask_fallback", "auto"),
        config.get("model_path", DEFAULT_LOCAL_QWEN_MODEL_PATH),
        config.get("backend", "local_qwen"),
        config.get("api_model", "gpt-4.1"),
        config.get("api_key"),
        config.get("base_url"),
        config.get("use_vlm_object_extraction", True),
    )
    if key not in _WORKER_OBJECT_EXTRACTORS:
        _WORKER_OBJECT_EXTRACTORS[key] = build_object_extractor(
            mode=config.get("mode", "reconstruct"),
            device=config.get("device", "cuda"),
            mask_fallback=config.get("mask_fallback", "auto"),
            model_path=config.get("model_path", DEFAULT_LOCAL_QWEN_MODEL_PATH),
            backend=config.get("backend", "local_qwen"),
            api_model=config.get("api_model", "gpt-4.1"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            use_vlm_object_extraction=config.get("use_vlm_object_extraction", True),
        )
    return _WORKER_OBJECT_EXTRACTORS[key]


def process_scene_with_agent_wrapper(args_tuple) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing that creates its own agent instance.
    
    Args:
        args_tuple: Tuple of (entry, api_key)
        
    Returns:
        Dictionary containing the complete pipeline results including type information
    """
    entry, config = args_tuple

    agent = Agent(
        api_key=config.get("api_key"),
        backend=config.get("backend", "local_qwen"),
        model_path=config.get("model_path", DEFAULT_LOCAL_QWEN_MODEL_PATH),
        api_model=config.get("api_model", "gpt-4.1"),
        code_model=config.get("code_model"),
        answer_model=config.get("answer_model"),
        base_url=config.get("base_url"),
        device=config.get("device", "cuda"),
    )

    extractor = get_worker_object_extractor(config)

    return process_scene_with_agent(
        entry,
        agent,
        mode=config.get("mode", "reconstruct"),
        mask_fallback=config.get("mask_fallback", "auto"),
        device=config.get("device", "cuda"),
        model_path=config.get("model_path", DEFAULT_LOCAL_QWEN_MODEL_PATH),
        api_model=config.get("api_model", "gpt-4.1"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        extract_output_dir=config.get("extract_output_dir"),
        force_extract=config.get("force_extract", False),
        extractor=extractor,
        use_vlm_object_extraction=config.get("use_vlm_object_extraction", True),
        max_code_repair_attempts=config.get("max_code_repair_attempts", 1),
    )



def generate_parse_execute_with_repair(agent: Agent, scene: Scene, max_code_repair_attempts: int = 1) -> Dict[str, Any]:
    generated_response = call_agent_with_retry(agent, 'generate_code', scene)
    parsed_code = agent.parse_LLM_response(scene, generated_response)
    initial_generated_response = generated_response
    initial_generated_code = parsed_code
    visual_clue = None
    parse_success = parsed_code is not None and parsed_code.strip() != ""
    execution_success = False
    code_repair_used = False
    code_repair_attempts = 0
    code_repair_error = None
    last_response = generated_response
    last_code = parsed_code
    last_error = None

    max_code_repair_attempts = max(0, int(max_code_repair_attempts or 0))

    while True:
        if parse_success:
            try:
                visual_clue = agent.execute(scene)
                execution_success = visual_clue != "there is an error during code generation, no visual clue provided"
                if execution_success:
                    return {
                        "generated_response": last_response,
                        "parsed_code": last_code,
                        "visual_clue": visual_clue,
                        "parse_success": True,
                        "execution_success": True,
                        "code_repair_used": code_repair_used,
                        "code_repair_attempts": code_repair_attempts,
                        "code_repair_error": code_repair_error,
                        "initial_generated_response": initial_generated_response,
                        "initial_generated_code": initial_generated_code,
                    }
                last_error = str(visual_clue)
            except Exception as exc:
                last_error = "Execution failed: {}\nTraceback: {}".format(exc, traceback.format_exc())
                execution_success = False
        else:
            last_error = "Code parsing failed: no ```python``` code block found in model response."

        code_repair_error = last_error
        if code_repair_attempts >= max_code_repair_attempts:
            return {
                "generated_response": last_response,
                "parsed_code": last_code,
                "visual_clue": visual_clue,
                "parse_success": parse_success,
                "execution_success": execution_success,
                "code_repair_used": code_repair_used,
                "code_repair_attempts": code_repair_attempts,
                "code_repair_error": code_repair_error,
                "initial_generated_response": initial_generated_response,
                "initial_generated_code": initial_generated_code,
            }

        code_repair_used = True
        code_repair_attempts += 1
        try:
            repair_response = call_agent_with_retry(
                agent,
                'repair_code',
                scene,
                previous_response=last_response,
                previous_code=last_code,
                error=last_error,
            )
        except Exception as exc:
            code_repair_error = "Code repair request failed: {}\nTraceback: {}".format(exc, traceback.format_exc())
            return {
                "generated_response": last_response,
                "parsed_code": last_code,
                "visual_clue": visual_clue,
                "parse_success": parse_success,
                "execution_success": False,
                "code_repair_used": code_repair_used,
                "code_repair_attempts": code_repair_attempts,
                "code_repair_error": code_repair_error,
                "initial_generated_response": initial_generated_response,
                "initial_generated_code": initial_generated_code,
            }

        last_response = repair_response
        last_code = agent.parse_LLM_response(scene, repair_response)
        parsed_code = last_code
        parse_success = parsed_code is not None and parsed_code.strip() != ""


def process_scene_with_agent(entry: Dict[str, Any], agent: Agent, mode: str = "reconstruct", mask_fallback: str = "auto", device: str = "cuda", model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH, api_model: str = "gpt-4.1", api_key: str = None, base_url: str = None, extract_output_dir: str = None, force_extract: bool = False, extractor=None, use_vlm_object_extraction: bool = True, max_code_repair_attempts: int = 1) -> Dict[str, Any]:
    """
    Process a single JSONL entry through the complete pipeline and extract type information.
    
    Args:
        entry: JSONL entry containing scene information
        agent: pySpatial Agent instance
        
    Returns:
        Dictionary containing the complete pipeline results including type information
    """
    scene_id = entry['id']
    question = entry.get('question', '')
    images = entry.get('images', [])
    expected_answer = entry.get('answer', entry.get('gt_answer'))
    answer_type = entry.get('answer_type')
    question_type = infer_question_type(question, expected_answer, answer_type)
    
    # Extract type from image paths
    scene_type = extract_type_from_images(images)

    
    scene = Scene(images, question, scene_id=scene_id)

    fallback_used = False
    generated_response = None
    parsed_code = None
    visual_clue = None
    generated_answer = None
    answer_reasoning = None
    answer_source = None
    answer_correct = False
    float_relative_error = None
    float_mra = None
    numeric_relative_error = None
    numeric_mra = None
    numeric_score = None
    execution_success = False
    answer_generation_success = False
    parse_success = False
    code_repair_used = False
    code_repair_attempts = 0
    code_repair_error = None
    initial_generated_code = None
    initial_generated_response = None
    try:
        if mode == "graph":
            pySpatial.extract_objects(
                scene,
                device=device,
                mask_fallback=mask_fallback,
                vlm_model_path=model_path,
                backend=agent.backend,
                api_model=api_model,
                api_key=api_key,
                base_url=base_url,
                save_dir=extract_output_dir,
                force_extract=force_extract,
                extractor=extractor,
                use_vlm_object_extraction=use_vlm_object_extraction,
            )
            pySpatial.build_graph(scene)
            question_type = infer_question_type(question, expected_answer, answer_type, scene.object_3d_boxes)

        # Step 1-2: Generate, parse, execute code, with optional repair retry.
        code_result = generate_parse_execute_with_repair(
            agent,
            scene,
            max_code_repair_attempts=max_code_repair_attempts,
        )
        generated_response = code_result["generated_response"]
        parsed_code = code_result["parsed_code"]
        visual_clue = code_result["visual_clue"]
        parse_success = code_result["parse_success"]
        execution_success = code_result["execution_success"]
        code_repair_used = code_result["code_repair_used"]
        code_repair_attempts = code_result["code_repair_attempts"]
        code_repair_error = code_result["code_repair_error"]
        initial_generated_response = code_result["initial_generated_response"]
        initial_generated_code = code_result["initial_generated_code"]

        # Step 3: Generate answer using visual clue (with retry)
        if execution_success:
            answer_response = call_agent_with_retry(agent, 'answer', scene, visual_clue)
            answer_generation_success = answer_response is not None

            if answer_generation_success:
                generated_answer = answer_response.answer
                answer_reasoning = getattr(answer_response, "reasoning", None)
                answer_source = "graph"

                # Step 4: Evaluate correctness
                if expected_answer is not None and generated_answer is not None:
                    answer_correct = evaluate_answer_correctness(generated_answer, expected_answer, answer_type)
                    numeric_metrics = compute_numeric_metrics(generated_answer, expected_answer, answer_type)
                    float_relative_error = numeric_metrics["float_relative_error"]
                    float_mra = numeric_metrics["float_mra"]
                    numeric_relative_error = numeric_metrics["numeric_relative_error"]
                    numeric_mra = numeric_metrics["numeric_mra"]
                    numeric_score = numeric_metrics["numeric_score"]

        # --- Fallback to basic QA if pySpatial pipeline didn't produce an answer ---
        if not answer_generation_success or generated_answer is None:
            print(f"[{scene_id}] pySpatial pipeline did not produce an answer, falling back to basic QA")
            fallback_response = call_agent_with_retry(agent, 'basic_qa', scene)
            if fallback_response is not None:
                fallback_used = True
                generated_answer = fallback_response.answer
                answer_reasoning = getattr(fallback_response, "reasoning", None)
                answer_source = "basic_qa"
                answer_generation_success = True
                if expected_answer is not None and generated_answer is not None:
                    answer_correct = evaluate_answer_correctness(generated_answer, expected_answer, answer_type)
                    numeric_metrics = compute_numeric_metrics(generated_answer, expected_answer, answer_type)
                    float_relative_error = numeric_metrics["float_relative_error"]
                    float_mra = numeric_metrics["float_mra"]
                    numeric_relative_error = numeric_metrics["numeric_relative_error"]
                    numeric_mra = numeric_metrics["numeric_mra"]
                    numeric_score = numeric_metrics["numeric_score"]

        result = {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "question": question,
            "images": images,
            "expected_answer": expected_answer,
            "answer_type": answer_type,
            "question_type": question_type,
            "parse_success": parse_success,
            "execution_success": execution_success,
            "answer_generation_success": answer_generation_success,
            "generated_answer": generated_answer,
            "answer_reasoning": answer_reasoning,
            "answer_source": answer_source,
            "answer_correct": answer_correct,
            "float_relative_error": float_relative_error,
            "float_mra": float_mra,
            "numeric_relative_error": numeric_relative_error,
            "numeric_mra": numeric_mra,
            "numeric_score": numeric_score if is_numeric_answer_type(answer_type, expected_answer) else (1.0 if answer_correct else 0.0),
            "generated_code": parsed_code,
            "generated_response": generated_response,
            "visual_clue": visual_clue,
            "object_3d_boxes": scene.object_3d_boxes,
            "fallback_used": fallback_used,
            "mode": mode,
            "mask_fallback": mask_fallback,
            "extract_output_dir": extract_output_dir,
            "force_extract": force_extract,
            "use_vlm_object_extraction": use_vlm_object_extraction,
            "code_repair_used": code_repair_used,
            "code_repair_attempts": code_repair_attempts,
            "code_repair_error": code_repair_error,
            "initial_generated_code": initial_generated_code,
            "initial_generated_response": initial_generated_response,
        }

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {scene_id}: {error_msg}")

        # Fallback to basic QA on complete pipeline failure
        fallback_answer = None
        fallback_correct = False
        fallback_float_relative_error = None
        fallback_float_mra = None
        fallback_numeric_relative_error = None
        fallback_numeric_mra = None
        fallback_numeric_score = None
        fallback_success = False
        try:
            print(f"[{scene_id}] Pipeline error, falling back to basic QA")
            fallback_response = call_agent_with_retry(agent, 'basic_qa', scene)
            if fallback_response is not None:
                fallback_answer = fallback_response.answer
                answer_reasoning = getattr(fallback_response, "reasoning", None)
                answer_source = "basic_qa"
                fallback_success = True
                fallback_used = True
                if expected_answer is not None and fallback_answer is not None:
                    fallback_correct = evaluate_answer_correctness(fallback_answer, expected_answer, answer_type)
                    fallback_numeric_metrics = compute_numeric_metrics(fallback_answer, expected_answer, answer_type)
                    fallback_float_relative_error = fallback_numeric_metrics["float_relative_error"]
                    fallback_float_mra = fallback_numeric_metrics["float_mra"]
                    fallback_numeric_relative_error = fallback_numeric_metrics["numeric_relative_error"]
                    fallback_numeric_mra = fallback_numeric_metrics["numeric_mra"]
                    fallback_numeric_score = fallback_numeric_metrics["numeric_score"]
        except Exception as fallback_e:
            print(f"[{scene_id}] Basic QA fallback also failed: {fallback_e}")

        return {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "question": question,
            "images": images,
            "expected_answer": expected_answer,
            "answer_type": answer_type,
            "question_type": question_type,
            "parse_success": False,
            "execution_success": False,
            "answer_generation_success": fallback_success,
            "generated_answer": fallback_answer,
            "answer_reasoning": answer_reasoning,
            "answer_source": answer_source,
            "answer_correct": fallback_correct,
            "float_relative_error": fallback_float_relative_error,
            "float_mra": fallback_float_mra,
            "numeric_relative_error": fallback_numeric_relative_error,
            "numeric_mra": fallback_numeric_mra,
            "numeric_score": fallback_numeric_score if is_numeric_answer_type(answer_type, expected_answer) else (1.0 if fallback_correct else 0.0),
            "generated_code": parsed_code,
            "generated_response": generated_response,
            "visual_clue": visual_clue,
            "object_3d_boxes": scene.object_3d_boxes,
            "fallback_used": fallback_used,
            "mode": mode,
            "mask_fallback": mask_fallback,
            "extract_output_dir": extract_output_dir,
            "force_extract": force_extract,
            "use_vlm_object_extraction": use_vlm_object_extraction,
            "code_repair_used": code_repair_used,
            "code_repair_attempts": code_repair_attempts,
            "code_repair_error": code_repair_error,
            "initial_generated_code": initial_generated_code,
            "initial_generated_response": initial_generated_response,
            "error": error_msg,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pySpatial Agent on MindCube dataset with type statistics")
    parser.add_argument("--jsonl_path", type=str, default=None,
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--dataset_json", type=str, default=None,
                       help="Path to Omni3D-Bench annotations.json")
    parser.add_argument("--image_root", type=str, default="/data/datasets/Omni3D-Bench/images",
                       help="Root directory for Omni3D-Bench images")
    parser.add_argument("--output_file", type=str,
                       default="pySpatial_mindcube_outputs",
                       help="Output directory for timestamped results and flowcharts")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of entries to process")
    parser.add_argument("--start_index", type=int, default=None,
                       help="Inclusive Omni3D question_index start, e.g. 400")
    parser.add_argument("--end_index", type=int, default=None,
                       help="Inclusive Omni3D question_index end, e.g. 500")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                       help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")
    parser.add_argument("--num_processes", type=int, default=1,
                       help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument("--disable_multiprocessing", action="store_true", 
                       help="Disable multiprocessing and run sequentially")
    parser.add_argument("--request_interval", type=float, default=0.1,
                       help="Minimum time between API requests in seconds (default: 0.1)")
    parser.add_argument("--filter_type", type=str, default=None,
                       choices=['among', 'around', 'rotation', 'unknown'],
                       help="Filter to only process specific scene type (among, around, rotation, or unknown)")
    parser.add_argument("--processed_dir", type=str, default=None,
                       help="Base directory for pre-processed scene data (optional)")
    parser.add_argument("--mode", choices=["reconstruct", "graph"], default="reconstruct",
                       help="Pipeline mode: legacy reconstruction or SpatialGraph")
    parser.add_argument("--mask_fallback", choices=["auto", "off"], default="auto",
                       help="Object extraction mask fallback mode used by graph mode")
    parser.add_argument("--backend", choices=["local_qwen", "openai"], default="local_qwen",
                       help="Model backend for code generation and answering")
    parser.add_argument("--model_path", type=str, default=DEFAULT_LOCAL_QWEN_MODEL_PATH,
                       help="Unified local model path for extraction, code generation, and answering")
    parser.add_argument("--local_model_path", type=str, default=None,
                       help="Deprecated alias for --model_path")
    parser.add_argument("--api_model", type=str, default="gpt-4.1",
                       help="Unified OpenAI model for extraction, code generation, and answering")
    parser.add_argument("--base_url", type=str, default=None,
                       help="OpenAI-compatible API base URL; falls back to CLOSEAI_BASE_URL or OPENAI_BASE_URL")
    parser.add_argument("--code_model", type=str, default=None,
                       help="Optional override for code generation model")
    parser.add_argument("--answer_model", type=str, default=None,
                       help="Optional override for answer model")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for local model and object extraction")
    parser.add_argument("--extract_output_dir", type=str, default=None,
                       help="Object extraction/debug output root; defaults to mask_fallback output root")
    parser.add_argument("--force_extract", action="store_true",
                       help="Always rerun object extraction instead of reading cached object_3d_positions.json")
    parser.add_argument("--use_vlm_object_extraction", action="store_true", default=True,
                       help="Enable APC-VLM-style VLM extraction of objects of interest")
    parser.add_argument("--no_vlm_object_extraction", action="store_false", dest="use_vlm_object_extraction",
                       help="Disable VLM object extraction and use rule-based question parsing")
    parser.add_argument("--max_code_repair_attempts", type=int, default=1,
                       help="Maximum code repair attempts after parse/execution failure; 0 disables repair")

    args = parser.parse_args()
    validate_index_range(args.start_index, args.end_index)
    
    # Update global rate limiting interval
    global min_request_interval
    min_request_interval = args.request_interval

    # Set the pre-processed scene base directory
    pySpatial.PROCESSED_BASE_DIR = args.processed_dir
    
    if args.local_model_path:
        args.model_path = args.local_model_path
    args.base_url = args.base_url or os.getenv("CLOSEAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if args.jsonl_path is None and args.dataset_json is None:
        args.dataset_json = "/data/datasets/Omni3D-Bench/annotations.json"
    if args.jsonl_path and not os.path.exists(args.jsonl_path):
        raise ValueError(f"JSONL file not found: {args.jsonl_path}")
    if args.dataset_json and not os.path.exists(args.dataset_json):
        raise ValueError(f"Dataset JSON file not found: {args.dataset_json}")
    
    # Determine number of processes
    if args.disable_multiprocessing:
        num_processes = 1
    else:
        num_processes = args.num_processes or cpu_count()
    
    print(f"Processing JSONL file: {args.jsonl_path or 'none'}")
    print(f"Processing dataset JSON: {args.dataset_json or 'none'}")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_file)
    output_dir = output_root / run_timestamp
    flowcharts_dir = output_dir / "flowcharts"
    flowcharts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {output_root}")
    print(f"Run output directory: {output_dir}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Start index: {args.start_index if args.start_index is not None else 'none'}")
    print(f"End index: {args.end_index if args.end_index is not None else 'none'}")
    print("Index range mode: question_index")
    print(f"Filter type: {args.filter_type or 'none (processing all types)'}")
    print(f"Number of processes: {num_processes}")
    print(f"Request interval: {min_request_interval}s")
    print(f"Mode: {args.mode}")
    print(f"Backend: {args.backend}")
    print(f"Base URL: {args.base_url or 'default OpenAI SDK'}")
    print(f"Mask fallback: {args.mask_fallback}")
    print(f"Extract output dir: {args.extract_output_dir or 'default mask_fallback output root'}")
    print(f"Force extract: {args.force_extract}")
    print(f"VLM object extraction: {args.use_vlm_object_extraction}")
    print(f"Max code repair attempts: {args.max_code_repair_attempts}")
    print("="*60)
    
    # Load all entries first
    range_enabled = args.start_index is not None or args.end_index is not None
    if args.dataset_json:
        entries = load_omni3d_entries(args.dataset_json, args.image_root)
    else:
        entries = []
        with open(args.jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not range_enabled and args.max_entries and len(entries) >= args.max_entries:
                    print(f"Reached maximum entries limit: {args.max_entries}")
                    break
                if not line.strip():
                    continue
                entries.append(json.loads(line.strip()))

    loaded_count = len(entries)
    entries = apply_index_range_and_max_entries(
        entries,
        start_index=args.start_index,
        end_index=args.end_index,
        max_entries=args.max_entries,
    )
    if range_enabled or args.max_entries is not None:
        print(f"Selected {len(entries)} entries for processing from {loaded_count} loaded entries")
    else:
        print(f"Loaded {len(entries)} entries for processing")

    if len(entries) == 0:
        print("No entries selected for processing. Exiting.")
        return

    # Filter entries by type if specified
    if args.filter_type:
        filtered_entries = []
        for entry in entries:
            images = entry.get('images', [])
            scene_type = extract_type_from_images(images)
            if scene_type == args.filter_type:
                filtered_entries.append(entry)

        print(f"Filtered to {len(filtered_entries)} entries of type '{args.filter_type}' (from {len(entries)} total)")
        entries = filtered_entries

        if len(entries) == 0:
            print(f"No entries found with type '{args.filter_type}'. Exiting.")
            return

    # Process entries
    start_time = time.time()
    
    if num_processes == 1 or args.disable_multiprocessing:
        # Sequential processing
        print("Running sequentially...")
        agent = Agent(
            api_key=args.api_key,
            backend=args.backend,
            model_path=args.model_path,
            api_model=args.api_model,
            code_model=args.code_model,
            answer_model=args.answer_model,
            base_url=args.base_url,
            device=args.device,
        )
        extractor = build_object_extractor(
            mode=args.mode,
            device=args.device,
            mask_fallback=args.mask_fallback,
            model_path=args.model_path,
            backend=args.backend,
            api_model=args.api_model,
            api_key=args.api_key,
            base_url=args.base_url,
            use_vlm_object_extraction=args.use_vlm_object_extraction,
        )
        results = []
        for i, entry in enumerate(entries, 1):
            print(f"Processing entry {i}/{len(entries)}: {entry.get('id', 'unknown')}")
            result = process_scene_with_agent(
                entry,
                agent,
                mode=args.mode,
                mask_fallback=args.mask_fallback,
                device=args.device,
                model_path=args.model_path,
                api_model=args.api_model,
                api_key=args.api_key,
                base_url=args.base_url,
                extract_output_dir=args.extract_output_dir,
                force_extract=args.force_extract,
                extractor=extractor,
                use_vlm_object_extraction=args.use_vlm_object_extraction,
                max_code_repair_attempts=args.max_code_repair_attempts,
            )
            results.append(result)
    else:
        # Multiprocessing
        print(f"Running with {num_processes} processes...")

        # Prepare arguments for multiprocessing
        worker_config = {
            "api_key": args.api_key,
            "backend": args.backend,
            "model_path": args.model_path,
            "api_model": args.api_model,
            "code_model": args.code_model,
            "answer_model": args.answer_model,
            "base_url": args.base_url,
            "mode": args.mode,
            "mask_fallback": args.mask_fallback,
            "device": args.device,
            "extract_output_dir": args.extract_output_dir,
            "force_extract": args.force_extract,
            "use_vlm_object_extraction": args.use_vlm_object_extraction,
            "max_code_repair_attempts": args.max_code_repair_attempts,
        }
        args_list = [(entry, worker_config) for entry in entries]

        pool = Pool(processes=num_processes, maxtasksperchild=4)
        async_result = pool.map_async(process_scene_with_agent_wrapper, args_list)
        results = async_result.get(timeout=3600)
        pool.terminate()
        pool.join()
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
    print(f"Average time per entry: {processing_time/len(entries):.2f} seconds")

    for result in results:
        flowchart_path = create_sample_flowchart(result, flowcharts_dir, extract_output_dir=args.extract_output_dir)
        if flowchart_path:
            result["flowchart_path"] = flowchart_path

    # Calculate statistics
    summary_stats = compute_summary_statistics(results)
    type_stats = summary_stats['type_stats']
    type_metrics = summary_stats['type_metrics']
    question_type_metrics = summary_stats['question_type_metrics']
    overall_stats = summary_stats['overall_stats']
    overall_metrics = summary_stats['overall_metrics']

    # Save results
    output_path = output_dir / "summary.json"

    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "run_timestamp": run_timestamp,
        "output_dir": str(output_dir),
        "flowcharts_dir": str(flowcharts_dir),
        "jsonl_source": args.jsonl_path,
        "dataset_json_source": args.dataset_json,
        "image_root": args.image_root,
        "processing_time_seconds": round(processing_time, 2),
        "avg_time_per_entry": round(processing_time/len(entries), 2),
        "num_processes_used": num_processes,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "index_range_mode": "question_index",
        "mode": args.mode,
        "backend": args.backend,
        "model_path": args.model_path,
        "api_model": args.api_model,
        "base_url": args.base_url,
        "mask_fallback": args.mask_fallback,
        "extract_output_dir": args.extract_output_dir,
        "force_extract": args.force_extract,
        "use_vlm_object_extraction": args.use_vlm_object_extraction,
        "max_code_repair_attempts": args.max_code_repair_attempts,
        "overall_metrics": overall_metrics,
        "type_metrics": type_metrics,
        "question_type_metrics": question_type_metrics,
        "raw_statistics": dict(type_stats),
        "results": [summarize_result_for_output(result) for result in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(make_json_safe(summary), f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n=== MindCube Evaluation Results ===")
    total = overall_metrics['total_count']
    print(f"Total entries processed: {total}")
    print(f"\n=== Overall Performance ===")
    print(f"Parse success: {overall_stats['parse_success']}/{total} ({overall_metrics['parse_rate']:.1f}%)")
    print(f"Execution success: {overall_stats['execution_success']}/{total} ({overall_metrics['execution_rate']:.1f}%)")
    print(f"Answer generation: {overall_stats['answer_generation_success']}/{total} ({overall_metrics['answer_generation_rate']:.1f}%)")
    print(f"Answer correctness: {overall_stats['correct_answers']}/{overall_stats['evaluable_answers']} ({overall_metrics['correctness_rate']:.1f}%)")
    print(f"MRA score: {overall_metrics['mra_score']:.4f} over {overall_metrics['mra_score_count']} samples")
    print(f"ACC_MRA: {overall_metrics['acc_mra']:.4f} over {overall_metrics['acc_mra_count']} samples")

    print(f"\n=== Statistics by Question Type ===")
    for question_type, metrics in question_type_metrics.items():
        print(f"\n{question_type.upper()}:")
        print(f"  Count: {metrics['count']}")
        print(f"  Parse rate: {metrics['parse_rate']:.1f}%")
        print(f"  Execution rate: {metrics['execution_rate']:.1f}%")
        print(f"  Answer generation rate: {metrics['answer_generation_rate']:.1f}%")
        print(f"  Correctness rate: {metrics['correctness_rate']:.1f}% ({metrics['correct_answers']}/{metrics['evaluable_count']})")
        if 'mra' in metrics:
            print(f"  Numeric MRA: {metrics.get('mra', 0.0):.4f} ({metrics.get('mra_count', 0)} samples)")
        print(f"  Error rate: {metrics['error_rate']:.1f}% ({metrics['error_count']}/{metrics['count']})")

    
    print(f"\n=== Statistics by Type ===")
    for scene_type, metrics in type_metrics.items():
        print(f"\n{scene_type.upper()}:")
        print(f"  Count: {metrics['count']}")
        print(f"  Parse rate: {metrics['parse_rate']:.1f}%")
        print(f"  Execution rate: {metrics['execution_rate']:.1f}%")
        print(f"  Answer generation rate: {metrics['answer_generation_rate']:.1f}%")
        print(f"  Correctness rate: {metrics['correctness_rate']:.1f}% ({type_stats[scene_type]['correct_answers']}/{metrics['evaluable_count']})")
        print(f"  Error rate: {metrics['error_rate']:.1f}% ({metrics['error_count']}/{metrics['count']})")


if __name__ == "__main__":
    # This guard is important for multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down...")
        sys.exit(1)








