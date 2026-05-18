"""Utility helpers for object_3d_extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union
import re

import numpy as np
from PIL import Image, ImageDraw


def load_rgb_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")
    return Image.open(image_path).convert("RGB")


def validate_object_names(object_names: Iterable[str]) -> List[str]:
    if object_names is None:
        raise ValueError("object_names must be a non-empty list of strings")

    names = [str(name).strip() for name in object_names if str(name).strip()]
    if not names:
        raise ValueError("object_names must contain at least one non-empty object name")
    return names


def extract_object_names_from_question_options(question: str) -> List[str]:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    names = []  # type: List[str]
    option_pattern = re.compile(
        r"(?:^|\s)([A-H])\.\s*(.*?)(?=\s+[A-H]\.\s*|$)",
        flags=re.IGNORECASE,
    )
    option_spans = [match.span() for match in option_pattern.finditer(question)]
    for _, option_text in option_pattern.findall(question):
        cleaned = _clean_object_name(option_text)
        if cleaned and not _is_non_object_option(cleaned):
            names.append(cleaned)

    question_without_options = question
    if option_spans:
        question_without_options = question[: option_spans[0][0]]

    reference_patterns = [
        r"\bshowing\s+the\s+(.+?)\s+from\b",
        r"\bshowing\s+a\s+(.+?)\s+from\b",
        r"\bof\s+the\s+(.+?)(?:\?|$|\s+in\s+image|\s+from\s+image|\s+is\b|\s+are\b)",
        r"\bof\s+a\s+(.+?)(?:\?|$|\s+in\s+image|\s+from\s+image|\s+is\b|\s+are\b)",
        r"\brelative\s+to\s+the\s+(.+?)(?:\?|$|\s+in\s+image|\s+from\s+image)",
    ]
    for pattern in reference_patterns:
        for match in re.findall(pattern, question_without_options, flags=re.IGNORECASE):
            cleaned = _clean_object_name(match)
            if cleaned and not _is_non_object_option(cleaned):
                names.insert(0, cleaned)

    return _dedupe_preserve_order(names)


def cxcywh_to_xyxy(box: np.ndarray, width: int, height: int) -> List[int]:
    cx, cy, bw, bh = [float(v) for v in box]
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return clamp_box([x1, y1, x2, y2], width, height)


def clamp_box(box: Iterable[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def bbox_wh(box2d: Iterable[int]) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box2d]
    return [max(0, x2 - x1), max(0, y2 - y1)]


def ensure_save_dir(save_dir: Optional[Union[str, Path]]) -> Path:
    if save_dir is None:
        save_dir = "outputs/debug"
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def save_debug_visuals(
    image: Image.Image,
    object_name: str,
    box2d: Optional[List[int]],
    mask: Optional[np.ndarray],
    save_dir: Optional[Union[str, Path]],
) -> None:
    save_path = ensure_save_dir(save_dir)
    safe_name = object_name.replace("/", "_").replace(" ", "_")

    debug_image = image.copy()
    draw = ImageDraw.Draw(debug_image)
    if box2d is not None:
        draw.rectangle(box2d, outline=(255, 0, 0), width=3)
        draw.text((box2d[0], max(0, box2d[1] - 12)), object_name, fill=(255, 0, 0))
    debug_image.save(save_path / f"{safe_name}_box.png")

    if mask is not None:
        mask_u8 = (mask.astype(np.float32) > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask_u8, mode="L").save(save_path / f"{safe_name}_mask.png")

        overlay = np.array(image).copy()
        overlay[mask_u8 > 0] = (0.5 * overlay[mask_u8 > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(save_path / f"{safe_name}_overlay.png")


def _clean_object_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`({\[]+", "", text)
    text = re.sub(r"[\"'`)}\].,;:!?]+$", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+(from different viewpoints|from different views)$", "", text)
    return text.strip()


def _is_non_object_option(text: str) -> bool:
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
        "true",
        "false",
        "clockwise",
        "counterclockwise",
        "same",
        "different",
    }
    return text in non_objects


def _dedupe_preserve_order(names: Iterable[str]) -> List[str]:
    seen = set()
    output = []  # type: List[str]
    for name in names:
        if name not in seen:
            seen.add(name)
            output.append(name)
    return output
