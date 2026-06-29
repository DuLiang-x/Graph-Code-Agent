#!/usr/bin/env python
"""VLM candidate scoring experiment for regenerating object_3d_positions.json.

This script reads cached object_3d_positions.json files, asks a VLM to score the
existing 2D detection candidates without detector scores, and can optionally
rerun SAM/depth/3D reconstruction using the VLM-selected candidate index.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

REPO_TEST_DIR = Path(__file__).resolve().parents[1]
if str(REPO_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_TEST_DIR))

from object_3d_extraction import Object3DExtractionConfig, Object3DLocator
from object_3d_extraction.object_3d_locator import clamp_box_to_image, draw_labeled_box
from scripts.demo_extract_3d_positions import (
    DEFAULT_LOCAL_QWEN_MODEL_PATH,
    OpenAIVLRefinementModel,
    QwenVLRefinementModel,
)

DEFAULT_INPUT_DIR = "/data/duliang/pySpatial-test/outputs/Omnitest5nomask"
DEFAULT_DATASET_JSON = "/data/datasets/Omni3D-Bench/annotations.json"
DEFAULT_OUTPUT_DIR = "/data/duliang/pySpatial-test/outputs/vlm_candidate_scoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cached GroundingDINO candidates with a VLM and optionally regenerate 3D boxes.")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--dataset_json", default=DEFAULT_DATASET_JSON)
    parser.add_argument("--sample_index", type=int, nargs="+")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--object_name")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["local_qwen", "openai"], default="local_qwen")
    parser.add_argument("--api_model", default="gpt-4.1")
    parser.add_argument("--api_key")
    parser.add_argument("--base_url")
    parser.add_argument("--vlm_model_path", default=DEFAULT_LOCAL_QWEN_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--regenerate_3d_boxes", action="store_true")
    parser.add_argument("--mask_fallback", default="auto", choices=["auto", "none"])
    parser.add_argument("--box_threshold", type=float, default=0.05)
    parser.add_argument("--text_threshold", type=float, default=0.05)
    parser.add_argument("--count_max_instances", type=int, default=20)
    return parser.parse_args()


def sample_index_from_name(name: str) -> Optional[int]:
    match = re.search(r"omni3d_(\d+)", name)
    return int(match.group(1)) if match else None


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("_") or "object"


def load_cached_record(json_path: Path) -> Tuple[str, Dict[str, Any]]:
    data = json.loads(json_path.read_text())
    if isinstance(data, dict) and len(data) == 1:
        sample_id = next(iter(data))
        record = data[sample_id]
        return sample_id, record
    sample_id = data.get("sample_id") or json_path.parent.name
    return sample_id, data


def iter_sample_jsons(input_dir: Path, args: argparse.Namespace) -> Iterable[Path]:
    sample_indices = set(args.sample_index or [])
    paths = sorted(input_dir.glob("omni3d_*/object_3d_positions.json"), key=lambda p: sample_index_from_name(p.parent.name) or -1)
    for path in paths:
        idx = sample_index_from_name(path.parent.name)
        if idx is None:
            continue
        if sample_indices and idx not in sample_indices:
            continue
        if not sample_indices and args.start_index is not None and idx < args.start_index:
            continue
        if not sample_indices and args.end_index is not None and idx > args.end_index:
            continue
        yield path


def load_question_map(dataset_json: str) -> Dict[int, str]:
    path = Path(dataset_json)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        entries = raw.get("data") or raw.get("annotations") or raw.get("samples") or raw.get("items") or list(raw.values())
    else:
        entries = raw
    mapping: Dict[int, str] = {}
    if isinstance(entries, dict):
        entries = list(entries.values())
    for offset, item in enumerate(entries or []):
        if not isinstance(item, dict):
            continue
        idx = item.get("question_index")
        if idx is None:
            idx = sample_index_from_name(str(item.get("id") or item.get("sample_id") or ""))
        if idx is None:
            idx = offset
        question = item.get("question") or item.get("query") or item.get("prompt")
        if question:
            mapping[int(idx)] = str(question)
    return mapping


def build_vlm_model(args: argparse.Namespace):
    if args.backend == "openai":
        return OpenAIVLRefinementModel(api_key=args.api_key, model=args.api_model, base_url=args.base_url)
    return QwenVLRefinementModel(args.vlm_model_path, device=args.device)


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
        label = str(candidate_index)
        x1, y1 = box[0], box[1]
        draw.rectangle([x1, y1, x1 + 42, y1 + 28], fill="black")
        draw.text((x1 + 8, y1 + 5), label, fill=color)
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


def make_all_selected_overlay(image: Image.Image, selections: Dict[str, Dict[str, Any]]) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "blue"]
    for idx, (object_name, result) in enumerate(selections.items()):
        candidate = result.get("selected_candidate")
        if not candidate:
            continue
        box = clamp_box(candidate["box2d"], image.size)
        color = colors[idx % len(colors)]
        draw_labeled_box(draw, box, color, width=5)
        label = object_name
        x1, y1 = box[0], box[1]
        label_w = max(60, min(360, 8 * len(label) + 16))
        draw.rectangle([x1, y1, x1 + label_w, y1 + 28], fill="black")
        draw.text((x1 + 8, y1 + 5), label, fill=color)
    return overlay


def make_final_3d_boxes_overlay(image: Image.Image, result: Dict[str, Dict[str, Any]]) -> Image.Image:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "blue"]
    drawn_index = 0
    for object_name, item in result.items():
        if not isinstance(item, dict):
            continue
        if "box2d" not in item:
            continue
        if item.get("error"):
            continue
        box = clamp_box(item["box2d"], image.size)
        color = colors[drawn_index % len(colors)]
        drawn_index += 1
        draw_labeled_box(draw, box, color, width=5)
        label = str(object_name)
        x1, y1 = box[0], box[1]
        label_w = max(60, min(420, 8 * len(label) + 16))
        draw.rectangle([x1, y1, x1 + label_w, y1 + 28], fill="black")
        draw.text((x1 + 8, y1 + 5), label, fill=color)
    return overlay


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


def score_object_candidates(
    vlm_model,
    image: Image.Image,
    sample_output_dir: Path,
    question: str,
    object_name: str,
    candidates: List[Dict[str, Any]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    object_dir = sample_output_dir / "candidate_scoring"
    object_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(object_name)
    overlay = make_candidate_overlay(image, candidates)
    grid = make_crop_grid(image, candidates)
    overlay_path = object_dir / f"{safe}_candidate_overlay.png"
    grid_path = object_dir / f"{safe}_crop_grid.png"
    overlay.save(overlay_path)
    grid.save(grid_path)

    prompt = build_scoring_prompt(question, object_name, candidates)
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
        selected_candidate = next((candidate for idx, candidate in enumerate(candidates) if int(candidate.get("candidate_index", idx)) == selected_index), None)
    selected_overlay_path = None
    if selected_candidate is not None:
        selected_overlay = make_candidate_overlay(image, candidates, selected_index=selected_index)
        selected_overlay_path = object_dir / f"{safe}_vlm_selected_overlay.png"
        selected_overlay.save(selected_overlay_path)
    return {
        "object_name": object_name,
        "selected_index": selected_index,
        "selected_candidate": selected_candidate,
        "raw_response": response,
        "parsed_response": parsed,
        "prompt": prompt,
        "candidate_overlay_path": str(overlay_path),
        "crop_grid_path": str(grid_path),
        "selected_overlay_path": str(selected_overlay_path) if selected_overlay_path else None,
    }


def object_candidates(record: Dict[str, Any], object_name: str) -> List[Dict[str, Any]]:
    result = record.get("result") or {}
    item = result.get(object_name) or {}
    if item.get("counting_target"):
        return []
    candidates = item.get("candidates_considered") or []
    normalized = []
    for idx, candidate in enumerate(candidates):
        if "box2d" not in candidate:
            continue
        copy = dict(candidate)
        copy["candidate_index"] = int(copy.get("candidate_index", idx))
        copy["box2d"] = [int(v) for v in copy["box2d"]]
        normalized.append(copy)
    return normalized


def build_regeneration_config(args: argparse.Namespace) -> Object3DExtractionConfig:
    config = Object3DExtractionConfig()
    config.detection.box_threshold = args.box_threshold
    config.detection.text_threshold = args.text_threshold
    config.detection.count_max_instances = args.count_max_instances
    config.detection.use_vlm_refinement = False
    config.mask_fallback_mode = args.mask_fallback
    return config


def regenerate_3d_boxes(
    locator: Object3DLocator,
    record: Dict[str, Any],
    question: str,
    sample_output_dir: Path,
    forced_selections: Dict[str, int],
) -> Dict[str, Any]:
    objects = list(record.get("objects") or (record.get("result") or {}).keys())
    return locator.extract(
        record["image"],
        objects,
        visualize=True,
        save_dir=sample_output_dir,
        question=question,
        answer=str(record.get("answer") or ""),
        question_type=record.get("question_type"),
        forced_candidate_selections=forced_selections,
    )


def process_sample(json_path: Path, args: argparse.Namespace, question_map: Dict[int, str], vlm_model, regeneration_locator: Optional[Object3DLocator] = None) -> Dict[str, Any]:
    sample_id, record = load_cached_record(json_path)
    sample_index = sample_index_from_name(sample_id) or sample_index_from_name(json_path.parent.name)
    question = record.get("question") or question_map.get(sample_index, "")
    image_path = Path(record["image"])
    image = Image.open(image_path).convert("RGB")
    sample_output_dir = Path(args.output_dir) / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    result = record.get("result") or {}
    objects = list(record.get("objects") or result.keys())
    if args.object_name:
        objects = [name for name in objects if name == args.object_name]

    if args.dry_run:
        print(f"[{sample_id}] image={image_path}")
        print(f"[{sample_id}] question={question}")
        for object_name in objects:
            candidates = object_candidates(record, object_name)
            final_idx = (result.get(object_name) or {}).get("final_selected_index")
            print(f"  - {object_name}: candidates={len(candidates)}, old_final_index={final_idx}")
        return {"sample_id": sample_id, "dry_run": True, "objects": objects}

    scoring_results: Dict[str, Any] = {}
    forced_selections: Dict[str, int] = {}
    for object_name in objects:
        candidates = object_candidates(record, object_name)
        if not candidates:
            scoring_results[object_name] = {"object_name": object_name, "error": "no_cached_candidates"}
            continue
        scoring = score_object_candidates(
            vlm_model,
            image,
            sample_output_dir,
            question,
            object_name,
            candidates,
            args.max_new_tokens,
        )
        scoring_results[object_name] = scoring
        if scoring.get("selected_index") is not None:
            forced_selections[object_name] = int(scoring["selected_index"])

    all_overlay = make_all_selected_overlay(image, scoring_results)
    all_overlay_path = sample_output_dir / f"{sample_id}_vlm_all_selected_overlay.png"
    all_overlay.save(all_overlay_path)

    scoring_json_path = sample_output_dir / f"{sample_id}_vlm_candidate_scores.json"
    serializable_scoring = json.loads(json.dumps(scoring_results, default=str))
    scoring_json_path.write_text(json.dumps(serializable_scoring, indent=2, ensure_ascii=False))

    output_record = dict(record)
    output_record["object_extraction_method"] = f"{record.get('object_extraction_method', 'unknown')}+vlm_candidate_scoring"
    output_record["vlm_candidate_scoring_source"] = str(json_path)
    output_record["vlm_candidate_scoring_results"] = serializable_scoring

    final_3d_boxes_overlay_path = None
    if args.regenerate_3d_boxes and not args.dry_run:
        if regeneration_locator is None:
            raise RuntimeError("regeneration_locator is required when --regenerate_3d_boxes is enabled")
        regenerated = regenerate_3d_boxes(regeneration_locator, record, question, sample_output_dir, forced_selections)
        final_overlay = make_final_3d_boxes_overlay(image, regenerated)
        final_overlay_path = sample_output_dir / f"{sample_id}_final_3d_boxes_overlay.png"
        final_overlay.save(final_overlay_path)
        final_3d_boxes_overlay_path = str(final_overlay_path)
        output_record["result"] = regenerated
        output_record["objects"] = list(record.get("objects") or objects)
        output_json_path = sample_output_dir / "object_3d_positions.json"
        output_json_path.write_text(json.dumps({sample_id: output_record}, indent=2, ensure_ascii=False))

    return {
        "sample_id": sample_id,
        "sample_index": sample_index,
        "objects_scored": len([item for item in scoring_results.values() if not item.get("error")]),
        "forced_candidate_selections": forced_selections,
        "all_selected_overlay_path": str(all_overlay_path),
        "final_3d_boxes_overlay_path": final_3d_boxes_overlay_path,
        "scoring_json_path": str(scoring_json_path),
        "regenerated": bool(args.regenerate_3d_boxes),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    question_map = load_question_map(args.dataset_json)
    sample_paths = list(iter_sample_jsons(input_dir, args))
    print(f"Found {len(sample_paths)} sample(s).")

    if args.dry_run:
        vlm_model = None
    else:
        vlm_model = build_vlm_model(args)

    regeneration_locator = None
    final_3d_boxes_overlay_path = None
    if args.regenerate_3d_boxes and not args.dry_run:
        print("Initializing Object3DLocator once for 3D regeneration...")
        regeneration_locator = Object3DLocator(
            config=build_regeneration_config(args),
            device=args.device,
            vlm_model=None,
        )

    summary = []
    for json_path in sample_paths:
        try:
            summary.append(process_sample(json_path, args, question_map, vlm_model, regeneration_locator=regeneration_locator))
        except Exception as exc:
            sample_id = json_path.parent.name
            print(f"[{sample_id}] ERROR: {exc}")
            summary.append({"sample_id": sample_id, "error": str(exc)})

    summary_path = output_dir / "vlm_candidate_scoring_summary.json"
    summary_path.write_text(json.dumps({"input_dir": str(input_dir), "output_dir": str(output_dir), "samples": summary}, indent=2, ensure_ascii=False))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
