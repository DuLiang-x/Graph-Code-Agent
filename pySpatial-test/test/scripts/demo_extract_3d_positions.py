#!/usr/bin/env python
"""CLI demo for object_3d_extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_TEST_DIR = Path(__file__).resolve().parents[1]
if str(REPO_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_TEST_DIR))

from object_3d_extraction import Object3DLocator
from object_3d_extraction.utils import extract_object_names_from_question_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 3D object positions from MindCube samples.")
    parser.add_argument("--image", default=None, help="Optional image path override.")
    parser.add_argument("--sample_json", default=None, help="Path to one JSON sample file.")
    parser.add_argument("--jsonl", default=None, help="Path to a MindCube JSONL file.")
    parser.add_argument("--sample_id", default=None, help="Sample id to select from --jsonl.")
    parser.add_argument("--sample_index", type=int, default=None, help="Sample index to select from --jsonl.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of JSONL samples to run.")
    parser.add_argument(
        "--base_data_path",
        default=None,
        help="Base path for relative sample image paths when --image is not provided.",
    )
    parser.add_argument("--device", default="cuda", help="Device for model inference, e.g. cuda or cpu.")
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
        samples = []
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                sample = json.loads(line)
                if args.sample_id is not None and sample.get("id") == args.sample_id:
                    return [sample]
                if args.sample_id is None and args.sample_index is not None and idx == args.sample_index:
                    return [sample]
                if args.sample_id is None and args.sample_index is None:
                    samples.append(sample)
                    if args.max_samples is not None and len(samples) >= args.max_samples:
                        break
        if args.sample_id is not None:
            raise ValueError("sample_id not found in jsonl: {}".format(args.sample_id))
        if args.sample_index is not None:
            raise ValueError("sample_index out of range: {}".format(args.sample_index))
        return samples

    return []


def resolve_image_path(args: argparse.Namespace, sample) -> str:
    if args.image:
        return args.image

    images = sample.get("images") or []
    if not images:
        raise ValueError("Sample does not contain images; please pass --image")

    image_path = Path(images[0])
    if image_path.is_absolute():
        return str(image_path)

    if not args.base_data_path:
        raise ValueError("Sample image path is relative; please pass --base_data_path")

    return str(Path(args.base_data_path) / image_path)


def resolve_object_names(sample) -> list:
    question = sample.get("question", "")
    object_names = extract_object_names_from_question_options(question)
    if not object_names:
        raise ValueError("Could not extract object names from sample question/options")
    return object_names


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
        raise ValueError("--sample_json or --jsonl must be provided")

    locator = Object3DLocator(device=args.device)
    output_root = Path(args.save_dir)
    written_files = []

    for idx, sample in enumerate(samples):
        sample_key = sample.get("id", "sample_{}".format(idx))
        image = resolve_image_path(args, sample)
        object_names = resolve_object_names(sample)
        print(
            "[{}/{}] {} objects: {}".format(
                idx + 1,
                len(samples),
                sample_key,
                json.dumps(object_names, ensure_ascii=False),
            )
        )

        sample_save_dir = output_root / sample_key
        try:
            result = locator.extract(
                image=image,
                object_names=object_names,
                visualize=not args.no_visualize,
                save_dir=sample_save_dir,
            )
            record = {
                "sample_id": sample_key,
                "image": image,
                "objects": object_names,
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
