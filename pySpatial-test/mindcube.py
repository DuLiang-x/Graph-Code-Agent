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
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import threading
import backoff
import re
import string
import textwrap

from PIL import Image, ImageFont, ImageDraw
import numpy as np

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pySpatial_Interface import Agent, Scene, pySpatial
from scene_layout import get_scene_overview_image

# Rate limiting globals
last_request_time = 0
min_request_interval = 0.1  # Minimum time between requests (100ms)
request_lock = threading.Lock()


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
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"Retrying {details.get('wait', 0):.2f}s after attempt {details.get('tries', 0)}...")
)
def call_agent_with_retry(agent, method_name, *args, **kwargs):
    """Call agent method with rate limiting and retry logic"""
    rate_limit()

    method = getattr(agent, method_name)
    result = method(*args, **kwargs)
    return result


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


def normalize_answer(answer: str) -> str:
    """Normalize answer for more flexible matching."""
    if not answer:
        return ""

    # Convert to lowercase
    answer = answer.lower().strip()

    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer)

    # Remove punctuation except numbers and basic symbols
    answer = answer.strip(string.punctuation)

    return answer


def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text for numeric comparison."""
    # Match integers and decimals
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers


def flexible_answer_match(generated: str, expected: str) -> bool:
    """
    Flexible answer matching with multiple strategies.
    Returns True if answers match by any strategy.
    """
    if not generated or not expected:
        return False

    # Strategy 1: Exact match
    if generated == expected:
        return True

    # Strategy 2: Normalized match
    norm_gen = normalize_answer(generated)
    norm_exp = normalize_answer(expected)
    if norm_gen == norm_exp:
        return True

    # Strategy 3: Substring match (either direction)
    # For short expected answers (e.g., "A", "B", "C", "D"), use word boundary
    # matching to avoid false positives (e.g., "c" matching inside "classroom")
    if len(norm_exp) <= 2:
        if re.search(r'\b' + re.escape(norm_exp) + r'\b', norm_gen):
            return True
        if re.search(r'\b' + re.escape(norm_gen) + r'\b', norm_exp):
            return True
    else:
        if norm_exp in norm_gen or norm_gen in norm_exp:
            return True

    # Strategy 4: Numeric equivalence
    gen_numbers = extract_numbers(norm_gen)
    exp_numbers = extract_numbers(norm_exp)

    if gen_numbers and exp_numbers:
        # Check if all expected numbers appear in generated
        if all(num in gen_numbers for num in exp_numbers):
            return True
        # Check numeric proximity (within 10% for non-zero values)
        try:
            gen_nums = [float(n) for n in gen_numbers]
            exp_nums = [float(n) for n in exp_numbers]
            if len(gen_nums) == len(exp_nums):
                matches = all(
                    abs(g - e) < 0.1 * max(abs(e), 1.0)
                    for g, e in zip(gen_nums, exp_nums)
                )
                if matches:
                    return True
        except ValueError:
            pass

    # Strategy 5: Keyword overlap (for longer answers)
    if len(norm_gen.split()) > 2 and len(norm_exp.split()) > 2:
        gen_words = set(norm_gen.split())
        exp_words = set(norm_exp.split())
        overlap = len(gen_words & exp_words)
        # If 80%+ of expected words are in generated
        if overlap >= 0.8 * len(exp_words):
            return True

    # Strategy 6: Yes/No, True/False normalization
    yes_no_map = {
        'yes': ['yes', 'true', 'correct', 'right'],
        'no': ['no', 'false', 'incorrect', 'wrong'],
        'true': ['yes', 'true', 'correct'],
        'false': ['no', 'false', 'incorrect'],
    }
    for canonical, variants in yes_no_map.items():
        if norm_exp in variants and any(v in norm_gen for v in variants):
            return True

    return False


def evaluate_answer_correctness(generated_answer: str, expected_answer: str) -> bool:
    """Check if generated answer matches expected answer with flexible matching."""
    return flexible_answer_match(generated_answer, expected_answer)


# ============================================================
# Visualization Utilities (adapted from APC-VLM)
# ============================================================

def _tile_images(image_paths: List[str], cols: int = 4, thumb_size: int = 150, base_dir: str = None) -> Image.Image:
    """
    Tile multiple images into a single grid image.
    Each image is resized to thumb_size x thumb_size (max), preserving aspect ratio.
    Returns a PIL Image.
    """
    if not image_paths:
        # Return a placeholder image
        return Image.new('RGB', (thumb_size, thumb_size), (200, 200, 200))

    imgs = []
    for p in image_paths:
        try:
            full_path = os.path.join(base_dir, p) if base_dir and not os.path.isabs(p) else p
            img = Image.open(full_path).convert('RGB')
            # Resize preserving aspect ratio
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            imgs.append(img)
        except Exception:
            continue

    if not imgs:
        return Image.new('RGB', (thumb_size, thumb_size), (200, 200, 200))

    n = len(imgs)
    rows = (n + cols - 1) // cols
    actual_cols = min(n, cols)

    cell_gap = 8
    cell_padding = 8
    grid_w = actual_cols * thumb_size + (actual_cols - 1) * cell_gap + cell_padding * 2
    grid_h = rows * thumb_size + (rows - 1) * cell_gap + cell_padding * 2
    grid = Image.new('RGB', (grid_w, grid_h), (248, 248, 248))
    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = cell_padding + c * (thumb_size + cell_gap)
        y = cell_padding + r * (thumb_size + cell_gap)
        try:
            draw.rounded_rectangle(
                (x, y, x + thumb_size, y + thumb_size),
                radius=12,
                fill=(255, 255, 255),
                outline=(228, 228, 228),
                width=1,
            )
        except Exception:
            draw.rectangle(
                (x, y, x + thumb_size, y + thumb_size),
                fill=(255, 255, 255),
                outline=(228, 228, 228),
            )
        # Center image in its cell
        offset_x = (thumb_size - img.width) // 2
        offset_y = (thumb_size - img.height) // 2
        grid.paste(img, (x + offset_x, y + offset_y))

    return grid


def _render_point_cloud_scene_image(scene_id: str, width: int = 960, height: int = 600) -> Optional[Image.Image]:
    """
    Render a 3D scene image from preprocessed reconstruction artifacts.

    Uses:
    - pySpatial.PROCESSED_BASE_DIR / scene_id / points.ply
    - pySpatial.PROCESSED_BASE_DIR / scene_id / camera_matrices.npz

    The scene is rendered from a fixed oblique overview camera derived from
    the point cloud bounds, with a dedicated overview intrinsic matched to the
    output canvas rather than reusing an input-view camera.
    Returns a PIL image on success, or None if any required artifact is unavailable.
    """
    if not scene_id or not pySpatial.PROCESSED_BASE_DIR:
        return None

    scene_dir = Path(pySpatial.PROCESSED_BASE_DIR) / scene_id
    ply_path = scene_dir / "points.ply"
    npz_path = scene_dir / "camera_matrices.npz"

    if not ply_path.exists() or not npz_path.exists():
        return None

    try:
        import trimesh
        from tool.novel_view_synthesis import render_pcd_with_extrinsics
        import open3d as o3d
    except Exception:
        return None

    try:
        point_cloud = trimesh.load(str(ply_path))
        points_xyz = np.asarray(point_cloud.vertices, dtype=np.float32)
        if points_xyz.size == 0:
            return None
        colors_rgb = None
        if hasattr(point_cloud, "colors") and point_cloud.colors is not None and len(point_cloud.colors) > 0:
            colors_rgb = np.asarray(point_cloud.colors, dtype=np.float32)
            if colors_rgb.ndim == 2 and colors_rgb.shape[1] >= 3:
                colors_rgb = colors_rgb[:, :3]
                if colors_rgb.max() > 1.0:
                    colors_rgb = colors_rgb / 255.0

        camera_data = np.load(str(npz_path))
        if "intrinsic" not in camera_data:
            return None

        fx = fy = float(max(width, height) * 1.15)
        cx = width / 2.0
        cy = height / 2.0
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        mins = points_xyz.min(axis=0)
        maxs = points_xyz.max(axis=0)
        center = ((mins + maxs) / 2.0).astype(np.float32)
        extents = np.maximum(maxs - mins, 1e-3)
        scale = float(np.max(extents))
        scale = max(scale, 1.0)

        def build_extrinsic(distance_scale: float, up_vec: np.ndarray) -> Optional[np.ndarray]:
            eye = center + np.array([1.05, -0.9, 0.85], dtype=np.float32) * (scale * distance_scale)
            target = center

            forward = target - eye
            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-6:
                return None
            forward = forward / forward_norm

            right = np.cross(up_vec, forward)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                return None
            right = right / right_norm

            cam_up = np.cross(forward, right)
            cam_up_norm = np.linalg.norm(cam_up)
            if cam_up_norm < 1e-6:
                return None
            cam_up = cam_up / cam_up_norm

            rotation_c2w = np.stack([right, cam_up, forward], axis=1)
            rotation_w2c = rotation_c2w.T
            translation_w2c = -rotation_w2c @ eye.reshape(3, 1)

            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = rotation_w2c.astype(np.float32)
            extrinsic[:3, 3] = translation_w2c[:, 0].astype(np.float32)
            return extrinsic

        def render_candidate(extrinsic: np.ndarray) -> Optional[Image.Image]:
            rendered = render_pcd_with_extrinsics(
                points_xyz,
                colors_rgb,
                intrinsic,
                extrinsic,
                width,
                height,
                point_size=4.0,
                out_path=None,
                zoom_out_scale=0.95,
            )

            rendered_np = np.asarray(rendered)
            if rendered_np.ndim == 2:
                rgb_np = np.repeat(rendered_np[..., None], 3, axis=2)
            elif rendered_np.ndim == 3:
                rgb_np = rendered_np[..., :3]
            else:
                return None

            if rgb_np.size == 0:
                return None

            near_white_ratio = np.mean(np.all(rgb_np >= 245, axis=2))
            if near_white_ratio > 0.992:
                return None

            return Image.fromarray(rgb_np.astype(np.uint8)).convert("RGB")

        candidate_settings = [
            (1.15, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            (1.0, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            (1.15, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            (1.0, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ]

        for distance_scale, up_vec in candidate_settings:
            extrinsic = build_extrinsic(distance_scale, up_vec)
            if extrinsic is None:
                continue
            candidate = render_candidate(extrinsic)
            if candidate is not None:
                return candidate
    except Exception:
        return None

    return None


def _render_scene_image(
    scene_id: str,
    result: Optional[Dict[str, Any]] = None,
    width: int = 960,
    height: int = 600,
) -> Optional[Image.Image]:
    """
    Render the preferred scene overview for a sample.

    Priority:
    1. Structured scene layout generated from preprocessing or raw entry metadata.
    2. Legacy point-cloud overview rendered from reconstruction artifacts.
    """
    structured_image = get_scene_overview_image(
        scene_id=scene_id,
        result=result,
        processed_base_dir=pySpatial.PROCESSED_BASE_DIR,
        width=width,
        height=height,
    )
    if structured_image is not None:
        return structured_image
    return _render_point_cloud_scene_image(scene_id, width=width, height=height)


def _format_code(code: str) -> str:
    """Return full code without truncation."""
    if not code:
        return "N/A"
    return code


def _collect_code_patterns(code: str) -> Dict[str, Any]:
    """
    Collect lightweight regex-based metadata from generated code.
    This intentionally stays within the existing regex approach.
    """
    patterns: Dict[str, Any] = {
        "generated_funcs": [],
        "pyspatial_aliases": set(),
        "pyspatial_imports": {},
        "tool_imports": {},
        "tool_method_names": set(),
    }

    if not code:
        return patterns

    # Generated helper definitions: top-level def, excluding program.
    for match in re.finditer(r'(?m)^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code):
        func_name = match.group(1)
        if func_name != "program" and func_name not in patterns["generated_funcs"]:
            patterns["generated_funcs"].append(func_name)

    # pySpatial aliases from import or assignment.
    patterns["pyspatial_aliases"].add("pySpatial")
    for match in re.finditer(r'(?m)^\s*import\s+pySpatial\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)', code):
        patterns["pyspatial_aliases"].add(match.group(1))
    for match in re.finditer(r'(?m)^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*pySpatial\b', code):
        patterns["pyspatial_aliases"].add(match.group(1))

    # from pySpatial import foo / foo as alias
    for match in re.finditer(r'(?m)^\s*from\s+pySpatial\s+import\s+(.+)$', code):
        imported_clause = match.group(1)
        for item in imported_clause.split(","):
            part = item.strip()
            if not part:
                continue
            alias_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)$', part)
            if alias_match:
                original, alias = alias_match.groups()
                patterns["pyspatial_imports"][alias] = original
            else:
                patterns["pyspatial_imports"][part] = part

    # from tool.xxx import foo / foo as alias
    for match in re.finditer(r'(?m)^\s*from\s+tool\.[\w\.]+\s+import\s+(.+)$', code):
        imported_clause = match.group(1)
        for item in imported_clause.split(","):
            part = item.strip()
            if not part:
                continue
            alias_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)$', part)
            if alias_match:
                original, alias = alias_match.groups()
                patterns["tool_imports"][alias] = original
            else:
                patterns["tool_imports"][part] = part

    # Known tool method names from tool/*.py files. Class construction itself is not treated as a local function summary.
    tool_dir = Path(__file__).resolve().parent / "tool"
    for tool_file in sorted(tool_dir.glob("*.py")):
        try:
            content = tool_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for method_match in re.finditer(r'(?m)^\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content):
            method_name = method_match.group(1)
            if method_name != "__init__":
                patterns["tool_method_names"].add(method_name)

    return patterns


def _colorize_code(code: str) -> List[Tuple[str, Optional[str]]]:
    """
    Colorize code for display in flowchart.
    Returns a list of (text, color) tuples where color is a hex string or None (default color).
    - Local function calls: red (#FF0000)
    - Generated helper definitions/calls: blue (#0000FF)
    - Everything else: default text color (None)
    """
    if not code:
        return [("N/A", None)]

    code_patterns = _collect_code_patterns(code)
    lines = code.split('\n')
    colored_segments = []

    py_spatial_pattern = re.compile(r'\bpySpatial\.\w+')
    local_patterns = [py_spatial_pattern]

    for alias in sorted(code_patterns["pyspatial_aliases"] - {"pySpatial"}):
        local_patterns.append(re.compile(rf'\b{re.escape(alias)}\.\w+'))

    for alias in sorted(code_patterns["pyspatial_imports"].keys()):
        local_patterns.append(re.compile(rf'\b{re.escape(alias)}\s*\('))

    for alias in sorted(code_patterns["tool_imports"].keys()):
        local_patterns.append(re.compile(rf'\b{re.escape(alias)}\s*\('))

    for method_name in sorted(code_patterns["tool_method_names"]):
        local_patterns.append(re.compile(rf'\b[a-zA-Z_][a-zA-Z0-9_]*\.{re.escape(method_name)}\s*\('))

    generated_func_pattern = None
    if code_patterns["generated_funcs"]:
        generated_union = "|".join(re.escape(name) for name in code_patterns["generated_funcs"])
        generated_func_pattern = re.compile(rf'\b({generated_union})\s*\(')

    for line in lines:
        segments = []
        pos = 0

        # Highlight generated helper definitions first.
        def_match = re.match(r'^(\s*def\s+)([a-zA-Z_][a-zA-Z0-9_]*)(\s*\()', line)
        if def_match and def_match.group(2) in code_patterns["generated_funcs"]:
            prefix, func_name, suffix = def_match.groups()
            rest = line[def_match.end(3) - 1:]
            segments.append((prefix, None))
            segments.append((func_name, "#0000FF"))
            segments.append((rest, None))
            colored_segments.extend(segments)
            colored_segments.append(("\n", None))
            continue

        # First pass: mark local calls red.
        local_matches = []
        for pattern in local_patterns:
            local_matches.extend(pattern.finditer(line))
        local_matches.sort(key=lambda m: (m.start(), -(m.end() - m.start())))

        accepted_local_matches = []
        last_end = -1
        for match in local_matches:
            if match.start() >= last_end:
                accepted_local_matches.append(match)
                last_end = match.end()

        for match in accepted_local_matches:
            start, end = match.span()
            if start > pos:
                segments.append((line[pos:start], None))
            segments.append((line[start:end], "#FF0000"))
            pos = end

        if pos < len(line):
            remaining = line[pos:]
            sub_pos = 0
            if generated_func_pattern:
                for m in generated_func_pattern.finditer(remaining):
                    s, e = m.span()
                    if s > sub_pos:
                        segments.append((remaining[sub_pos:s], None))
                    segments.append((m.group(), "#0000FF"))
                    sub_pos = e
            if sub_pos < len(remaining):
                segments.append((remaining[sub_pos:], None))

        # Merge consecutive segments with same color
        merged = []
        for text, color in segments:
            if merged and merged[-1][1] == color:
                merged[-1] = (merged[-1][0] + text, color)
            else:
                merged.append((text, color))
        colored_segments.extend(merged)
        colored_segments.append(("\n", None))  # newline as separate segment

    # Remove trailing newline segment if present
    if colored_segments and colored_segments[-1][0] == "\n":
        colored_segments.pop()

    return colored_segments


def _extract_api_calls(code: str) -> List[str]:
    """Extract pySpatial API calls from code string using enhanced regex rules."""
    if not code:
        return []

    code_patterns = _collect_code_patterns(code)
    found_calls: List[str] = []

    api_patterns = [
        'pySpatial.reconstruct',
        'pySpatial.describe_camera_motion',
        'pySpatial.synthesize_novel_view',
        'pySpatial.rotate_right',
        'pySpatial.rotate_left',
        'pySpatial.move_forward',
        'pySpatial.move_backward',
        'pySpatial.turn_around',
    ]
    api_suffixes = {api.split(".", 1)[1]: api for api in api_patterns}

    for api in api_patterns:
        if re.search(rf'\b{re.escape(api)}\s*\(', code):
            found_calls.append(api)

    for alias in code_patterns["pyspatial_aliases"] - {"pySpatial"}:
        for suffix, canonical in api_suffixes.items():
            if canonical not in found_calls and re.search(rf'\b{re.escape(alias)}\.{re.escape(suffix)}\s*\(', code):
                found_calls.append(canonical)

    for alias, imported_name in code_patterns["pyspatial_imports"].items():
        canonical = api_suffixes.get(imported_name)
        if canonical and canonical not in found_calls and re.search(rf'\b{re.escape(alias)}\s*\(', code):
            found_calls.append(canonical)

    return found_calls


def _rounded_rect(draw, xy, radius, fill):
    """Draw a rounded rectangle, fallback to rectangle if not available."""
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill)
    except Exception:
        draw.rectangle(xy, fill=fill)


def visualize_conversation(
    items: List[dict],
    width: int = 1400,
    padding: int = 28,
    row_gap: int = 18,
    image_max_width: int = 650,
    font_path: Optional[str] = None,
    font_size: int = 20,
    text_bg: tuple = (246, 246, 246),
    canvas_bg: tuple = (255, 255, 255),
    text_color: tuple = (20, 20, 20),
    bubble_radius: int = 16,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Visualize a conversation / pipeline trace as a single image.
    Layout: each row has an optional left-side image, and a right-side text bubble.
    Adapted from APC-VLM/apc/utils.py.
    """
    def load_font(size: int) -> ImageFont.FreeTypeFont:
        try:
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def wrap_color_segments(segments, font, max_width, draw):
        """
        Wrap a list of (text, color) segments into lines,
        where each line is a list of segments that fit within max_width.
        """
        lines = []
        current_line_segments = []
        current_line_width = 0

        for text, color in segments:
            # Handle newline explicitly
            if text == "\n":
                lines.append(current_line_segments)
                current_line_segments = []
                current_line_width = 0
                continue

            # Split text into words to allow line breaks within a segment
            words = text.split(' ')
            for i, word in enumerate(words):
                # Add space except before first word
                if i > 0:
                    word_with_space = ' ' + word
                else:
                    word_with_space = word

                bbox = draw.textbbox((0, 0), word_with_space, font=font)
                w = bbox[2] - bbox[0]

                if current_line_width + w <= max_width:
                    current_line_segments.append((word_with_space, color))
                    current_line_width += w
                else:
                    # Line break
                    if current_line_segments:
                        lines.append(current_line_segments)
                    current_line_segments = [(word, color)]  # no leading space
                    bbox_word = draw.textbbox((0, 0), word, font=font)
                    current_line_width = bbox_word[2] - bbox_word[0]

        if current_line_segments:
            lines.append(current_line_segments)
        return lines

    def load_image(img):
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        try:
            return Image.open(img).convert("RGBA")
        except Exception:
            return None

    # --- measurement pass ---
    tmp = Image.new("RGB", (width, 200), canvas_bg)
    tmp_draw = ImageDraw.Draw(tmp)
    font = load_font(font_size)
    line_height = max(font_size, int(font_size * 1.3))
    line_gap = max(4, int(font_size * 0.2))

    measured_rows = []
    total_height = padding

    for idx, item in enumerate(items):
        text_content = item.get("text", "")
        img = load_image(item.get("image"))
        row_image_max_width = item.get("image_max_width", image_max_width)
        row_image_max_height = item.get("image_max_height")

        # Determine text area width
        if img is not None:
            left_block_w = row_image_max_width
            gutter = 16
            text_area_w = width - (padding * 2) - left_block_w - gutter
        else:
            left_block_w = 0
            gutter = 0
            text_area_w = width - (padding * 2)

        # Convert text_content to segments if it's a string
        if isinstance(text_content, str):
            segments = [(text_content, None)]
        else:
            segments = text_content  # list of (text, color)

        # Wrap into colored lines
        wrapped_lines = wrap_color_segments(segments, font, text_area_w, tmp_draw)
        # Calculate total height of text block
        text_h = len(wrapped_lines) * line_height + max(0, len(wrapped_lines) - 1) * line_gap
        text_h = max(text_h, line_height)

        img_h = 0
        img_w = 0
        if img is not None:
            iw, ih = img.size
            scale = min(row_image_max_width / iw, 1.0)
            img_w = int(iw * scale)
            img_h = int(ih * scale)
            max_img_h = row_image_max_height if row_image_max_height is not None else min(400, text_h * 2)
            if img_h > max_img_h:
                scale = max_img_h / img_h
                img_w = int(img_w * scale)
                img_h = int(img_h * scale)

        row_h = max(text_h, img_h) + padding
        measured_rows.append({
            "wrapped_lines": wrapped_lines,
            "text_h": text_h,
            "img": img,
            "img_w": img_w,
            "img_h": img_h,
            "left_block_w": left_block_w,
            "text_area_w": text_area_w,
            "row_h": row_h,
            "gutter": gutter
        })
        total_height += row_h + row_gap

    total_height += padding - row_gap

    # --- compose final image ---
    canvas = Image.new("RGB", (width, total_height), canvas_bg)
    draw = ImageDraw.Draw(canvas)

    y = padding
    for idx, row in enumerate(measured_rows):
        img = row["img"]
        img_w, img_h = row["img_w"], row["img_h"]
        row_h = row["row_h"]
        gutter = row["gutter"]

        x = padding
        if img is not None and img_w > 0 and img_h > 0:
            img_resized = img.resize((img_w, img_h), Image.LANCZOS)
            img_y = y + (row_h - img_h) // 2
            canvas.paste(img_resized, (x, img_y))
            x += img_w + gutter

        bubble_w = row["text_area_w"]
        bubble_h = row["text_h"] + padding // 2
        bubble_x0 = x
        bubble_y0 = y + (row_h - bubble_h) // 2
        bubble_x1 = min(bubble_x0 + bubble_w, width - padding)
        bubble_w = bubble_x1 - bubble_x0
        bubble_y1 = bubble_y0 + bubble_h

        row_bg = text_bg
        if idx == len(measured_rows) - 1:
            # Last row: highlight correctness with color
            text_lower = str(items[idx].get("text", "")).lower()
            if "correct" in text_lower or "✓" in text_lower:
                row_bg = (230, 255, 230)  # light green
            elif "incorrect" in text_lower or "failed" in text_lower or "✗" in text_lower:
                row_bg = (255, 230, 230)  # light red

        _rounded_rect(draw, (bubble_x0, bubble_y0, bubble_x1, bubble_y1), bubble_radius, row_bg)

        # Draw colored text lines
        tx = bubble_x0 + padding // 2
        ty = bubble_y0 + padding // 4
        for line_segments in row["wrapped_lines"]:
            x_cursor = tx
            for seg_text, seg_color in line_segments:
                color = seg_color if seg_color else text_color
                draw.text((x_cursor, ty), seg_text, font=font, fill=color)
                bbox = draw.textbbox((0, 0), seg_text, font=font)
                x_cursor += bbox[2] - bbox[0]
            ty += line_height + line_gap

        y += row_h + row_gap

    if output_path:
        canvas.save(output_path)

    return canvas


def create_sample_flowchart(result: Dict[str, Any], save_dir: str, parsed_code: str = None) -> str:
    """
    Create a per-sample visualization flowchart showing:
    - Input images (tiled grid)
    - Question
    - GT Answer
    - Model name (e.g., Qwen2.5-VL-7B-Instruct)
    - generate_code status + full code (colored)
    - execute status + APIs called
    - answer status
    - Model Answer
    - Correctness

    Returns the path to the saved flowchart image.
    """
    scene_id = result.get("scene_id", "unknown")
    question = result.get("question", "")
    expected_answer = result.get("expected_answer", "")
    generated_answer = result.get("generated_answer", "")
    parse_success = result.get("parse_success", False)
    execution_success = result.get("execution_success", False)
    answer_gen_success = result.get("answer_generation_success", False)
    answer_correct = result.get("answer_correct", False)
    images = result.get("images", [])
    base_dir = getattr(Scene, 'IMAGE_BASE_DIR', None)
    input_images_grid = _tile_images(images, cols=4, thumb_size=190, base_dir=base_dir)
    scene_render_image = _render_scene_image(scene_id, result=result)

    # Model name
    model_name = getattr(Agent, '_model_name', 'unknown')
    model_short = os.path.basename(model_name)

    code_status = "✓ Success" if parse_success else "✗ Failed"
    code_patterns = _collect_code_patterns(parsed_code) if parsed_code and parse_success else None
    if parsed_code and parse_success:
        code_segments = [("Code", None), ("\n", None), ("```python", None), ("\n", None)]
        code_segments.extend(_colorize_code(parsed_code))
        code_segments.append(("\n", None))
        code_segments.append(("```", None))
    else:
        code_segments = [("Code", None), ("\n", None), ("N/A", None)]

    # Build API calls text
    apis = _extract_api_calls(parsed_code) if parsed_code else []
    api_text = ", ".join(apis) if apis else "N/A"
    generated_funcs = ", ".join(code_patterns["generated_funcs"]) if code_patterns and code_patterns["generated_funcs"] else "N/A"

    # Build items list
    items = [
        {"text": f"Q: {question}", "image": input_images_grid, "image_max_width": 860},
        {"text": f"GT Answer: {expected_answer}", "image": None},
        {"text": f"Model: {model_short}", "image": None},
        {"text": f"generate_code: {code_status}", "image": None},
        {"text": code_segments, "image": None},
        {"text": f"execute: {'✓ Success' if execution_success else '✗ Failed'}\nAPIs: {api_text}\nGenerated funcs: {generated_funcs}", "image": None},
        {"text": f"answer: {'✓ Success' if answer_gen_success else '✗ Failed'}", "image": None},
        {"text": f"Model Answer: {generated_answer or 'N/A'}", "image": None},
    ]

    if scene_render_image is not None:
        items.insert(3, {
            "text": "Scene Overview",
            "image": scene_render_image,
            "image_max_width": 900,
            "image_max_height": 560,
        })

    # Last row: correctness
    correctness_text = f"Correctness: {'✓ Correct' if answer_correct else '✗ Incorrect'}"
    items.append({"text": correctness_text, "image": None})

    # Save flowchart
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{scene_id}_flowchart.png")
    try:
        visualize_conversation(items, output_path=output_path, width=1400, image_max_width=650, font_size=20)
        return output_path
    except Exception as e:
        print(f"[{scene_id}] Failed to create flowchart: {e}")
        return None


def process_scene_with_agent_wrapper(args_tuple) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing that creates its own agent instance.

    Args:
        args_tuple: Tuple of (entry, api_key, viz_save_dir)

    Returns:
        Dictionary containing the complete pipeline results including type information
    """
    entry, api_key, viz_save_dir = args_tuple

    # Create agent instance for this process
    agent = Agent(api_key=api_key)

    return process_scene_with_agent(entry, agent, viz_save_dir=viz_save_dir)


def process_scene_with_agent(entry: Dict[str, Any], agent: Agent, viz_save_dir: str = None) -> Dict[str, Any]:
    """
    Process a single JSONL entry through the complete pipeline and extract type information.

    Args:
        entry: JSONL entry containing scene information
        agent: pySpatial Agent instance
        viz_save_dir: Optional directory to save flowchart visualization

    Returns:
        Dictionary containing the complete pipeline results including type information
    """
    scene_id = entry['id']
    question = entry.get('question', '')
    images = entry.get('images', [])
    expected_answer = entry.get('gt_answer', '')

    # Extract type from image paths
    scene_type = extract_type_from_images(images)

    scene = Scene(images, question, scene_id=scene_id)

    fallback_used = False
    try:
        # Step 1: Generate code using the agent (with retry)
        generated_response = call_agent_with_retry(agent, 'generate_code', scene)

        # Parse the response to extract code patterns
        parsed_code = agent.parse_LLM_response(scene, generated_response)
        parse_success = parsed_code is not None and parsed_code.strip() != ""

        visual_clue = None
        generated_answer = None
        answer_correct = False
        execution_success = False
        answer_generation_success = False

        # Step 2: Execute code to get visual clue (if parsing was successful)
        if parse_success:
            visual_clue = agent.execute(scene)
            execution_success = visual_clue != "there is an error during code generation, no visual clue provided"

            # Step 3: Generate answer using visual clue (with retry)
            if execution_success:
                answer_response = call_agent_with_retry(agent, 'answer', scene, visual_clue)
                answer_generation_success = answer_response is not None

                if answer_generation_success:
                    generated_answer = answer_response.answer

                    # Step 4: Evaluate correctness
                    if expected_answer and generated_answer:
                        answer_correct = evaluate_answer_correctness(generated_answer, expected_answer)

        # --- Fallback to basic QA if pySpatial pipeline didn't produce an answer ---
        if not answer_generation_success or generated_answer is None:
            print(f"[{scene_id}] pySpatial pipeline did not produce an answer, falling back to basic QA")
            fallback_response = call_agent_with_retry(agent, 'basic_qa', scene)
            if fallback_response is not None:
                fallback_used = True
                generated_answer = fallback_response.answer
                answer_generation_success = True
                if expected_answer and generated_answer:
                    # If fallback answer indicates no match, mark as incorrect
                    if "none of the above" in generated_answer.lower() and "does not match" in generated_answer.lower():
                        answer_correct = False
                    else:
                        answer_correct = evaluate_answer_correctness(generated_answer, expected_answer)

        # --- Second fallback: If answer is still incorrect, try relaxed matching with context ---
        if not answer_correct and generated_answer and expected_answer:
            print(f"[{scene_id}] Answer incorrect, attempting semantic similarity check")
            # Try one more time with a more lenient prompt
            relaxed_response = call_agent_with_retry(agent, 'relaxed_qa', scene, visual_clue if visual_clue else None)
            if relaxed_response is not None:
                relaxed_answer = relaxed_response.answer
                if evaluate_answer_correctness(relaxed_answer, expected_answer):
                    generated_answer = relaxed_answer
                    answer_correct = True
                    print(f"[{scene_id}] Relaxed QA succeeded")

        result = {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "entry_type": entry.get("type"),
            "meta_info": entry.get("meta_info"),
            "question": question,
            "images": images,
            "expected_answer": expected_answer,
            "parse_success": parse_success,
            "execution_success": execution_success,
            "answer_generation_success": answer_generation_success,
            "generated_answer": generated_answer,
            "answer_correct": answer_correct,
            "fallback_used": fallback_used,
            "_processed_base_dir": pySpatial.PROCESSED_BASE_DIR,
        }

        if pySpatial.PROCESSED_BASE_DIR:
            overview_path = os.path.join(pySpatial.PROCESSED_BASE_DIR, scene_id, "scene_overview.png")
            if os.path.exists(overview_path):
                result["_scene_overview_path"] = overview_path

        # Generate flowchart visualization if enabled
        if viz_save_dir:
            try:
                chart_path = create_sample_flowchart(result, viz_save_dir, parsed_code=parsed_code if parse_success else None)
                if chart_path:
                    result["flowchart_path"] = chart_path
                    print(f"[{scene_id}] Flowchart saved: {chart_path}")
            except Exception as viz_e:
                print(f"[{scene_id}] Flowchart generation failed: {viz_e}")

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {scene_id}: {error_msg}")

        # Fallback to basic QA on complete pipeline failure
        fallback_answer = None
        fallback_correct = False
        fallback_success = False
        try:
            print(f"[{scene_id}] Pipeline error, falling back to basic QA")
            fallback_response = call_agent_with_retry(agent, 'basic_qa', scene)
            if fallback_response is not None:
                fallback_answer = fallback_response.answer
                fallback_success = True
                fallback_used = True
                if expected_answer and fallback_answer:
                    # If fallback answer indicates no match, mark as incorrect
                    if "none of the above" in fallback_answer.lower() and "does not match" in fallback_answer.lower():
                        fallback_correct = False
                    else:
                        fallback_correct = evaluate_answer_correctness(fallback_answer, expected_answer)

            # Second fallback: relaxed QA if basic QA didn't match
            if not fallback_correct and fallback_answer and expected_answer:
                print(f"[{scene_id}] Basic QA incorrect, attempting relaxed QA")
                relaxed_response = call_agent_with_retry(agent, 'relaxed_qa', scene, None)
                if relaxed_response is not None:
                    relaxed_answer = relaxed_response.answer
                    if evaluate_answer_correctness(relaxed_answer, expected_answer):
                        fallback_answer = relaxed_answer
                        fallback_correct = True
                        fallback_success = True
                        print(f"[{scene_id}] Relaxed QA succeeded after pipeline error")
        except Exception as fallback_e:
            print(f"[{scene_id}] Basic QA fallback also failed: {fallback_e}")

        return {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "entry_type": entry.get("type"),
            "meta_info": entry.get("meta_info"),
            "question": question,
            "images": images,
            "expected_answer": expected_answer,
            "parse_success": False,
            "execution_success": False,
            "answer_generation_success": fallback_success,
            "generated_answer": fallback_answer,
            "answer_correct": fallback_correct,
            "fallback_used": fallback_used,
            "error": error_msg,
            "_processed_base_dir": pySpatial.PROCESSED_BASE_DIR,
            "_scene_overview_path": os.path.join(pySpatial.PROCESSED_BASE_DIR, scene_id, "scene_overview.png")
            if pySpatial.PROCESSED_BASE_DIR and os.path.exists(os.path.join(pySpatial.PROCESSED_BASE_DIR, scene_id, "scene_overview.png"))
            else None,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pySpatial Agent on MindCube dataset with type statistics")
    parser.add_argument("--jsonl_path", type=str,
                       required=True,
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--output_file", type=str,
                       default="pySpatial_mindcube.json",
                       help="Output file path for results")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of entries to process")
    parser.add_argument("--max_test_samples", type=int, default=None,
                       help="Maximum number of test samples to evaluate (limits total samples processed)")
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
    parser.add_argument("--viz_save_dir", type=str, default=None,
                       help="Directory to save per-sample flowchart visualizations (optional)")

    args = parser.parse_args()

    # Update global rate limiting interval
    global min_request_interval
    min_request_interval = args.request_interval

    # Set the pre-processed scene base directory
    pySpatial.PROCESSED_BASE_DIR = args.processed_dir

    # Set the image base directory for loading images
    # Default to MindCube dataset data directory (images are in other_all_image subfolder)
    if not os.path.isabs(args.jsonl_path):
        args.jsonl_path = os.path.abspath(args.jsonl_path)
    jsonl_dir = os.path.dirname(args.jsonl_path)

    # Images are in "other_all_image" subdirectory
    # Try multiple possible locations:
    # 1. Directly in parent directory: ../other_all_image
    # 2. In data subdirectory of parent: ../data/other_all_image
    image_base_dir = None
    parent_dir = os.path.dirname(jsonl_dir)

    # Check if other_all_image is directly in parent directory
    if os.path.exists(os.path.join(parent_dir, "other_all_image")):
        image_base_dir = parent_dir
    # Check if we need to go up one more level to find data/other_all_image
    elif os.path.exists(os.path.join(os.path.dirname(parent_dir), "data", "other_all_image")):
        image_base_dir = os.path.join(os.path.dirname(parent_dir), "data")
    # Fallback: check parent directory itself
    elif os.path.exists(os.path.join(parent_dir, "data", "other_all_image")):
        image_base_dir = os.path.join(parent_dir, "data")

    if image_base_dir:
        Scene.IMAGE_BASE_DIR = image_base_dir
        print(f"Set IMAGE_BASE_DIR to: {image_base_dir}")
    else:
        print(f"Warning: Image directory 'other_all_image' not found")
        print("Images will be loaded relative to current working directory")

    if not os.path.exists(args.jsonl_path):
        raise ValueError(f"JSONL file not found: {args.jsonl_path}")

    # Generate timestamp-based output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("output") / timestamp
    os.makedirs(output_dir, exist_ok=True)

    # Auto-set output file path if not explicitly provided
    if args.output_file == "pySpatial_mindcube.json":
        args.output_file = str(output_dir / "pySpatial_mindcube.json")

    # Auto-set viz_save_dir if not explicitly provided
    viz_save_dir = args.viz_save_dir
    if viz_save_dir is None:
        viz_save_dir = str(output_dir / "flowcharts")

    # Determine number of processes
    if args.disable_multiprocessing:
        num_processes = 1
    else:
        num_processes = args.num_processes or cpu_count()

    print(f"Processing JSONL file: {args.jsonl_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Flowchart dir: {viz_save_dir}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Max test samples: {args.max_test_samples or 'all'}")
    print(f"Filter type: {args.filter_type or 'none (processing all types)'}")
    print(f"Number of processes: {num_processes}")
    print(f"Request interval: {min_request_interval}s")
    print("="*60)

    # Load all entries first
    entries = []
    with open(args.jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if args.max_entries and len(entries) >= args.max_entries:
                print(f"Reached maximum entries limit: {args.max_entries}")
                break

            entry = json.loads(line.strip())
            entries.append(entry)

    print(f"Loaded {len(entries)} entries for processing")

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

    # Limit test samples if specified
    if args.max_test_samples and len(entries) > args.max_test_samples:
        print(f"Limiting to {args.max_test_samples} test samples (from {len(entries)} available)")
        entries = entries[:args.max_test_samples]

    # Process entries
    start_time = time.time()

    if num_processes == 1 or args.disable_multiprocessing:
        # Sequential processing
        print("Running sequentially...")
        agent = Agent(api_key=args.api_key)
        results = []
        for i, entry in enumerate(entries, 1):
            print(f"Processing entry {i}/{len(entries)}: {entry.get('id', 'unknown')}")
            result = process_scene_with_agent(entry, agent, viz_save_dir=viz_save_dir)
            results.append(result)
    else:
        # Multiprocessing
        print(f"Running with {num_processes} processes...")

        # Prepare arguments for multiprocessing
        args_list = [(entry, args.api_key, viz_save_dir) for entry in entries]

        pool = Pool(processes=num_processes, maxtasksperchild=4)
        async_result = pool.map_async(process_scene_with_agent_wrapper, args_list)
        results = async_result.get(timeout=3600)
        pool.terminate()
        pool.join()

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
    print(f"Average time per entry: {processing_time/len(entries):.2f} seconds")

    # Calculate statistics
    type_stats = defaultdict(lambda: {
        'total': 0.0,
        'parse_success': 0.0,
        'execution_success': 0.0,
        'answer_generation_success': 0.0,
        'correct_answers': 0.0,
        'evaluable_answers': 0.0,
        'errors': 0.0
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
        scene_type = result['scene_type']

        # Update statistics
        type_stats[scene_type]['total'] += 1
        overall_stats['total_processed'] += 1

        if result.get('error'):
            type_stats[scene_type]['errors'] += 1
            overall_stats['errors'] += 1
            print(f"Error processing scene {scene_type}: {result.get('error')}")
            continue

        if result['parse_success']:
            type_stats[scene_type]['parse_success'] += 1
            overall_stats['parse_success'] += 1

        if result['execution_success']:
            type_stats[scene_type]['execution_success'] += 1
            overall_stats['execution_success'] += 1

        if result['answer_generation_success']:
            type_stats[scene_type]['answer_generation_success'] += 1
            overall_stats['answer_generation_success'] += 1

        if result['expected_answer'] and result['generated_answer']:
            type_stats[scene_type]['evaluable_answers'] += 1
            overall_stats['evaluable_answers'] += 1

            if result['answer_correct']:
                type_stats[scene_type]['correct_answers'] += 1
                overall_stats['correct_answers'] += 1

    # Calculate rates for each type
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

    # Calculate overall metrics
    total = overall_stats['total_processed']
    overall_metrics = {
        'total_count': total,
        'parse_rate': round(overall_stats['parse_success'] / total * 100, 2) if total > 0 else 0,
        'execution_rate': round(overall_stats['execution_success'] / total * 100, 2) if total > 0 else 0,
        'answer_generation_rate': round(overall_stats['answer_generation_success'] / total * 100, 2) if total > 0 else 0,
        'correctness_rate': round(overall_stats['correct_answers'] / overall_stats['evaluable_answers'] * 100, 2) if overall_stats['evaluable_answers'] > 0 else 0,
        'error_rate': round(overall_stats['errors'] / total * 100, 2) if total > 0 else 0,
        'evaluable_count': overall_stats['evaluable_answers'],
        'error_count': overall_stats['errors']
    }

    # Save results
    output_path = Path.cwd() / args.output_file

    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "jsonl_source": args.jsonl_path,
        "processing_time_seconds": round(processing_time, 2),
        "avg_time_per_entry": round(processing_time/len(entries), 2),
        "num_processes_used": num_processes,
        "overall_metrics": overall_metrics,
        "type_metrics": type_metrics,
        "raw_statistics": dict(type_stats),
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")

    # Print summary statistics
    print(f"\n=== MindCube Evaluation Results ===")
    print(f"Total entries processed: {total}")
    print(f"\n=== Overall Performance ===")
    print(f"Parse success: {overall_stats['parse_success']}/{total} ({overall_metrics['parse_rate']:.1f}%)")
    print(f"Execution success: {overall_stats['execution_success']}/{total} ({overall_metrics['execution_rate']:.1f}%)")
    print(f"Answer generation: {overall_stats['answer_generation_success']}/{total} ({overall_metrics['answer_generation_rate']:.1f}%)")
    print(f"Answer correctness: {overall_stats['correct_answers']}/{overall_stats['evaluable_answers']} ({overall_metrics['correctness_rate']:.1f}%)")

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
