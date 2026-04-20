import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


GRID_SIZE = 10
WORLD_UNIT = 1.45

_DIR_TO_VEC = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "front": (0, -1),
    "back": (0, 1),
    "inner": (0, -1),
    "outer": (0, 1),
}


def extract_scene_type(images: Optional[List[str]] = None, scene_id: str = "") -> str:
    for image_path in images or []:
        lowered = image_path.lower()
        if "among" in lowered:
            return "among"
        if "around" in lowered:
            return "around"
        if "rotation" in lowered:
            return "rotation"

    lowered_scene_id = (scene_id or "").lower()
    if "among" in lowered_scene_id:
        return "among"
    if "around" in lowered_scene_id:
        return "around"
    if "rotation" in lowered_scene_id:
        return "rotation"
    return "unknown"


def _normalize_orientation(value: Any) -> Optional[str]:
    mapping = {
        "face": "down",
        "back": "up",
        "left": "left",
        "right": "right",
        "front": "front",
        "down": "down",
        "up": "up",
        "inner": "inner",
        "outer": "outer",
        "none": None,
        "null": None,
        "": None,
        None: None,
    }
    if isinstance(value, str):
        return mapping.get(value.strip().lower(), value.strip().lower() or None)
    return mapping.get(value, None)


def _extract_around_image_names(images: List[str]) -> List[int]:
    image_names = []
    for image in images:
        filename = os.path.basename(image)
        match = re.search(r"(\d+)_frame(?:_[^.]+)?\.(?:png|jpg|jpeg)", filename)
        if not match:
            raise ValueError(f"Could not extract around image index from {image}")
        number = int(match.group(1))
        if number == 33:
            number = 3
        image_names.append(number)
    return image_names


def _gen_node(name: str, position: List[int], facing: Optional[str] = None, kind: str = "object") -> Optional[Dict[str, Any]]:
    if not name:
        return None
    node = {
        "name": name,
        "position": [int(position[0]), int(position[1])],
        "kind": kind,
    }
    if facing:
        node["facing"] = facing
    return node


def _infer_object_render_style(name: str, is_main: bool) -> Tuple[float, float]:
    lowered = str(name or "").strip().lower()
    role_scale = 1.0 if is_main else 0.82

    if any(token in lowered for token in ("bottle", "vase", "can", "lamp", "thermos", "man", "person", "woman")):
        return 0.56 * role_scale, 1.95 * role_scale
    if any(token in lowered for token in ("book", "plate", "phone", "mouse", "wallet", "remote", "keyboard")):
        return 0.95 * role_scale, 0.58 * role_scale
    if any(token in lowered for token in ("box", "cube", "speaker", "microwave", "toaster", "monitor")):
        return 0.9 * role_scale, 1.02 * role_scale
    return 0.82 * role_scale, (1.4 if is_main else 1.05)


def _apply_render_defaults(layout: Dict[str, Any]) -> Dict[str, Any]:
    objects = layout.get("objects", [])
    main_object = layout.get("main_object")

    for index, obj in enumerate(objects):
        is_main = obj.get("name") == main_object or (main_object is None and index == 0)
        render_scale, render_height = _infer_object_render_style(obj.get("name"), is_main)
        obj.setdefault("render_role", "main" if is_main else "secondary")
        obj.setdefault("render_scale", round(float(render_scale), 3))
        obj.setdefault("render_height", round(float(render_height), 3))

    return layout


def _build_among_layout(entry: Dict[str, Any], images: List[str], question: str) -> Dict[str, Any]:
    meta_info = entry.get("meta_info") or [[], []]
    objects = meta_info[0] if len(meta_info) > 0 and isinstance(meta_info[0], list) else []
    objects_orientation = meta_info[1] if len(meta_info) > 1 and isinstance(meta_info[1], list) else []

    while len(objects) < 5:
        objects.append("")
    while len(objects_orientation) < 5:
        objects_orientation.append(None)

    image_names = [os.path.basename(image).split("_")[0].lower() for image in images]
    recognizable = ["front", "left", "right", "back"]
    if all(name in recognizable for name in image_names):
        processed_image_names = image_names
    else:
        default_directions = ["front", "left", "back", "right"]
        processed_image_names = default_directions[: len(image_names)]

    view_base_name = "Image" if "image" in question.lower() else "View"
    local_view_map = [
        (f"{view_base_name} {index + 1}", global_view)
        for index, global_view in enumerate(processed_image_names)
    ]

    mapping_view_to_coordinates = {
        "front": [5, 6],
        "left": [4, 5],
        "right": [6, 5],
        "back": [5, 4],
    }
    facing_mapping = {
        "front": "up",
        "left": "right",
        "right": "left",
        "back": "down",
    }

    object_coordinates = [
        _gen_node(objects[0], [5, 5], _normalize_orientation(objects_orientation[0])),
        _gen_node(objects[1], [5, 8], _normalize_orientation(objects_orientation[1])),
        _gen_node(objects[2], [2, 5], _normalize_orientation(objects_orientation[2])),
        _gen_node(objects[3], [5, 2], _normalize_orientation(objects_orientation[3])),
        _gen_node(objects[4], [8, 5], _normalize_orientation(objects_orientation[4])),
    ]
    object_coordinates = [obj for obj in object_coordinates if obj is not None]

    view_coordinates = []
    for local_view_name, global_view_name in local_view_map:
        if global_view_name not in mapping_view_to_coordinates:
            continue
        view_coordinates.append(
            _gen_node(
                local_view_name,
                mapping_view_to_coordinates[global_view_name],
                facing_mapping[global_view_name],
                kind="view",
            )
        )

    main_object = object_coordinates[0]["name"] if object_coordinates else None
    main_facing = object_coordinates[0].get("facing") if object_coordinates else None
    return {
        "scene_type": "among",
        "orientation_mode": "object",
        "main_object": main_object,
        "main_object_facing": main_facing,
        "objects": object_coordinates,
        "views": view_coordinates,
    }


def _build_around_layout(entry: Dict[str, Any], images: List[str], question: str) -> Dict[str, Any]:
    meta_info = entry.get("meta_info") or []
    if len(meta_info) < 2:
        return {
            "scene_type": "around",
            "orientation_mode": "view",
            "objects": [],
            "views": [],
        }

    image_group_num = meta_info[0][0]
    object_len = meta_info[1][0]
    objects = list(meta_info[1][1])
    objects_orientation = list(meta_info[1][2])

    if len(objects) != object_len or len(objects_orientation) != object_len:
        return {
            "scene_type": "around",
            "orientation_mode": "view",
            "objects": [],
            "views": [],
        }

    item_id = entry.get("id", "")
    datasource = item_id.split("_")[0].replace("around", "")
    datasource = "self" if datasource == "new" else (datasource or "self")

    if image_group_num == 3 and datasource == "self":
        global_views = {1: "front", 2: "left", 3: "right"}
    elif image_group_num == 3 and datasource == "dl3dv10k":
        global_views = {1: "front", 2: "left", 3: "right", 4: "back"}
    elif image_group_num == 4:
        global_views = {1: "front", 2: "left", 3: "right", 4: "back"}
    elif image_group_num == 5:
        global_views = {1: "front", 2: "left", 3: "right", 4: "left", 5: "right"}
    elif image_group_num == 6 and datasource == "self":
        global_views = {1: "front", 2: "left", 3: "right", 4: "left", 5: "right", 6: "back"}
    elif image_group_num == 6 and datasource == "dl3dv10k":
        global_views = {1: "front", 2: "left", 3: "right", 4: "front", 5: "left", 6: "right"}
    else:
        global_views = {1: "front", 2: "left", 3: "right", 4: "back"}

    question_image_ids = _extract_around_image_names(images)
    view_base_name = "Image" if "image" in question.lower() else "View"
    local_view_map = [
        (f"{view_base_name} {index + 1}", global_views.get(global_id, "front"))
        for index, global_id in enumerate(question_image_ids)
    ]

    mapping_view_to_coordinates = {
        "front": [[5, 6], [5, 6], [5, 6]],
        "left": [[3, 5], [3, 5], [2, 5]],
        "right": [[6, 5], [7, 5], [7, 5]],
        "back": [[5, 4], [5, 4], [5, 4]],
    }
    facing_mapping = {
        "front": "up",
        "left": "right",
        "right": "left",
        "back": "down",
    }

    if object_len == 2:
        positions = [[4, 5], [5, 5]]
    elif object_len == 3:
        positions = [[4, 5], [5, 5], [6, 5]]
    elif object_len == 4:
        positions = [[3, 5], [4, 5], [5, 5], [6, 5]]
    else:
        positions = [[5, 5] for _ in objects]

    object_coordinates = []
    for obj_name, obj_pos, obj_facing in zip(objects, positions, objects_orientation):
        object_coordinates.append(_gen_node(obj_name, obj_pos, _normalize_orientation(obj_facing)))
    object_coordinates = [obj for obj in object_coordinates if obj is not None]

    mapping_index = max(0, min(object_len - 2, 2))
    view_coordinates = []
    for local_view_name, global_view in local_view_map:
        coords = mapping_view_to_coordinates.get(global_view, mapping_view_to_coordinates["front"])[mapping_index]
        view_coordinates.append(
            _gen_node(local_view_name, coords, facing_mapping.get(global_view), kind="view")
        )

    return {
        "scene_type": "around",
        "orientation_mode": "view",
        "main_object": object_coordinates[0]["name"] if object_coordinates else None,
        "main_object_facing": object_coordinates[0].get("facing") if object_coordinates else None,
        "objects": object_coordinates,
        "views": view_coordinates,
    }


def _build_rotation_layout(entry: Dict[str, Any], question: str) -> Dict[str, Any]:
    rotation_type = entry.get("type", "")
    objects = list(entry.get("meta_info", []) or [])
    view_base_name = "Image" if "image" in question.lower() else "View"

    if "two" in str(rotation_type):
        if rotation_type == "two_view_clockwise":
            object_positions = [[5, 3], [7, 5]]
            views = [
                _gen_node(f"{view_base_name} 1", [5, 5], "up", kind="view"),
                _gen_node(f"{view_base_name} 2", [5, 5], "right", kind="view"),
            ]
        elif rotation_type == "two_view_counterclockwise":
            object_positions = [[5, 3], [3, 5]]
            views = [
                _gen_node(f"{view_base_name} 1", [5, 5], "up", kind="view"),
                _gen_node(f"{view_base_name} 2", [5, 5], "left", kind="view"),
            ]
        else:
            object_positions = [[5, 3], [5, 7]]
            views = [
                _gen_node(f"{view_base_name} 1", [5, 5], "up", kind="view"),
                _gen_node(f"{view_base_name} 2", [5, 5], "down", kind="view"),
            ]
    elif "three" in str(rotation_type):
        object_positions = [[3, 5], [5, 3], [7, 5]]
        views = [
            _gen_node(f"{view_base_name} 1", [5, 5], "left", kind="view"),
            _gen_node(f"{view_base_name} 2", [5, 5], "up", kind="view"),
            _gen_node(f"{view_base_name} 3", [5, 5], "right", kind="view"),
        ]
    else:
        object_positions = [[3, 5], [5, 3], [7, 5], [5, 7]]
        views = [
            _gen_node(f"{view_base_name} 1", [5, 5], "left", kind="view"),
            _gen_node(f"{view_base_name} 2", [5, 5], "up", kind="view"),
            _gen_node(f"{view_base_name} 3", [5, 5], "right", kind="view"),
            _gen_node(f"{view_base_name} 4", [5, 5], "down", kind="view"),
        ]

    object_coordinates = []
    for obj_name, obj_pos in zip(objects, object_positions):
        object_coordinates.append(_gen_node(str(obj_name), obj_pos))
    object_coordinates = [obj for obj in object_coordinates if obj is not None]

    return {
        "scene_type": "rotation",
        "orientation_mode": "view",
        "main_object": object_coordinates[0]["name"] if object_coordinates else None,
        "main_object_facing": None,
        "objects": object_coordinates,
        "views": [view for view in views if view is not None],
    }


def _build_fallback_layout(entry: Dict[str, Any], scene_type: str) -> Dict[str, Any]:
    images = entry.get("images") or []
    question = entry.get("question", "")
    view_base_name = "Image" if "image" in question.lower() else "View"
    objects = []

    main_object = None
    match = re.search(r"showing the (.+?) from different viewpoints", question, re.IGNORECASE)
    if match:
        main_object = match.group(1).strip()
        objects.append(_gen_node(main_object, [5, 5]))

    views = []
    for index, image_path in enumerate(images):
        name = os.path.basename(image_path).split("_")[0].lower()
        if name not in {"front", "left", "right", "back"}:
            continue
        position_map = {
            "front": [5, 6],
            "left": [4, 5],
            "right": [6, 5],
            "back": [5, 4],
        }
        facing_map = {
            "front": "up",
            "left": "right",
            "right": "left",
            "back": "down",
        }
        views.append(_gen_node(f"{view_base_name} {index + 1}", position_map[name], facing_map[name], kind="view"))

    return {
        "scene_type": scene_type,
        "orientation_mode": "view" if scene_type in {"around", "rotation"} else "object",
        "main_object": main_object,
        "main_object_facing": None,
        "objects": [obj for obj in objects if obj is not None],
        "views": [view for view in views if view is not None],
    }


def build_scene_layout(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entry:
        return None

    images = list(entry.get("images") or [])
    scene_id = entry.get("id") or entry.get("scene_id") or ""
    question = entry.get("question", "")
    scene_type = extract_scene_type(images=images, scene_id=scene_id)

    try:
        if scene_type == "among":
            layout = _build_among_layout(entry, images, question)
        elif scene_type == "around":
            layout = _build_around_layout(entry, images, question)
        elif scene_type == "rotation":
            layout = _build_rotation_layout(entry, question)
        else:
            layout = _build_fallback_layout(entry, scene_type)
    except Exception:
        layout = _build_fallback_layout(entry, scene_type)

    layout["scene_id"] = scene_id
    layout["question"] = question
    layout["images"] = images
    layout["entry_type"] = entry.get("type")
    return _apply_render_defaults(layout)


def _get_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _grid_to_xy(position: List[int], board_left: int, board_top: int, board_size: int) -> Tuple[float, float]:
    cell = board_size / GRID_SIZE
    x = board_left + (position[0] + 0.5) * cell
    y = board_top + (position[1] + 0.5) * cell
    return x, y


def _grid_to_world(position: List[int]) -> Tuple[float, float, float]:
    return (
        (float(position[0]) - 4.5) * WORLD_UNIT,
        0.0,
        (float(GRID_SIZE - 1 - position[1])) * WORLD_UNIT,
    )


def _vec_add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_mul(a: Tuple[float, float, float], scalar: float) -> Tuple[float, float, float]:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = _norm(a)
    if length < 1e-8:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _make_camera(width: int, height: int) -> Dict[str, Any]:
    eye = (-7.2, 6.0, -6.8)
    target = (0.0, 1.0, 5.8)
    world_up = (0.0, 1.0, 0.0)
    forward = _normalize(_vec_sub(target, eye))
    right = _normalize(_cross(forward, world_up))
    up = _normalize(_cross(right, forward))
    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": up,
        "cx": width * 0.5,
        "cy": height * 0.57,
        "focal": min(width, height) * 1.05,
    }


def _project_point(
    point: Tuple[float, float, float],
    camera: Dict[str, Any],
) -> Optional[Tuple[float, float, float]]:
    rel = _vec_sub(point, camera["eye"])
    cam_x = _dot(rel, camera["right"])
    cam_y = _dot(rel, camera["up"])
    cam_z = _dot(rel, camera["forward"])
    if cam_z <= 0.2:
        return None
    x = camera["cx"] + camera["focal"] * (cam_x / cam_z)
    y = camera["cy"] - camera["focal"] * (cam_y / cam_z)
    return (x, y, cam_z)


def _draw_gradient_background(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    top = (245, 245, 243)
    bottom = (232, 228, 220)
    for row in range(height):
        blend = row / max(1, height - 1)
        color = tuple(
            int(top[channel] * (1.0 - blend) + bottom[channel] * blend)
            for channel in range(3)
        )
        draw.line((0, row, width, row), fill=color)


def _draw_ground_plane(
    draw: ImageDraw.ImageDraw,
    camera: Dict[str, Any],
    width: int,
    height: int,
) -> None:
    plane_corners_world = [
        (-7.5, 0.0, -1.5),
        (7.5, 0.0, -1.5),
        (8.8, 0.0, 14.5),
        (-8.8, 0.0, 14.5),
    ]
    plane_corners = [_project_point(point, camera) for point in plane_corners_world]
    if any(point is None for point in plane_corners):
        return

    plane_polygon = [(point[0], point[1]) for point in plane_corners if point is not None]
    draw.polygon(plane_polygon, fill=(250, 248, 244, 255), outline=(206, 202, 192, 255))

    for grid_index in range(-5, 7):
        start = _project_point((grid_index * WORLD_UNIT, 0.0, -1.5), camera)
        end = _project_point((grid_index * WORLD_UNIT, 0.0, 14.5), camera)
        if start is not None and end is not None:
            draw.line((start[0], start[1], end[0], end[1]), fill=(212, 208, 199, 215), width=2)

    for depth_index in range(-1, 11):
        z_value = depth_index * WORLD_UNIT
        start = _project_point((-7.5, 0.0, z_value), camera)
        end = _project_point((7.5, 0.0, z_value), camera)
        if start is not None and end is not None:
            draw.line((start[0], start[1], end[0], end[1]), fill=(216, 212, 203, 210), width=2)

    horizon = _project_point((0.0, 0.0, 14.0), camera)
    if horizon is not None:
        draw.line((0, horizon[1], width, horizon[1]), fill=(220, 216, 210, 128), width=1)


def _direction_to_ground_vector(direction: Optional[str], mode: str) -> Tuple[float, float, float]:
    normalized = _normalize_orientation(direction)
    if mode == "object":
        mapping = {
            "front": (0.0, 0.0, -1.0),
            "back": (0.0, 0.0, 1.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
            "inner": (0.0, 0.0, -1.0),
            "outer": (0.0, 0.0, 1.0),
        }
    else:
        mapping = {
            "up": (0.0, 0.0, 1.0),
            "front": (0.0, 0.0, 1.0),
            "inner": (0.0, 0.0, 1.0),
            "down": (0.0, 0.0, -1.0),
            "back": (0.0, 0.0, -1.0),
            "outer": (0.0, 0.0, -1.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
        }
    return mapping.get(normalized, (0.0, 0.0, 1.0))


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    lines = str(text or "").splitlines() or [""]
    if hasattr(draw, "multiline_textbbox"):
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=2)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    max_width = 0
    line_height = 0
    for line in lines:
        width, height = draw.textsize(line, font=font)
        max_width = max(max_width, width)
        line_height = max(line_height, height)
    total_height = line_height * len(lines) + max(0, len(lines) - 1) * 2
    return max_width, total_height


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor: Tuple[float, float],
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    background: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    text_w, text_h = _text_size(draw, text, font)
    padding_x = 10
    padding_y = 6
    bbox = (
        anchor[0],
        anchor[1],
        anchor[0] + text_w + padding_x * 2,
        anchor[1] + text_h + padding_y * 2,
    )
    _draw_rounded_rectangle(
        draw,
        bbox,
        radius=10,
        fill=background,
        outline=(255, 255, 255, 140),
        width=1,
    )
    text_anchor = (anchor[0] + padding_x, anchor[1] + padding_y - 1)
    if "\n" in text and hasattr(draw, "multiline_text"):
        draw.multiline_text(text_anchor, text, font=font, fill=fill, spacing=2)
    else:
        draw.text(text_anchor, text, font=font, fill=fill)
    return bbox


def _expand_box(
    box: Tuple[float, float, float, float],
    padding_x: float,
    padding_y: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    pad_y = padding_x if padding_y is None else padding_y
    return (
        box[0] - padding_x,
        box[1] - pad_y,
        box[2] + padding_x,
        box[3] + pad_y,
    )


def _boxes_overlap(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
    padding: float = 0.0,
) -> bool:
    return not (
        box_a[2] + padding <= box_b[0]
        or box_b[2] + padding <= box_a[0]
        or box_a[3] + padding <= box_b[1]
        or box_b[3] + padding <= box_a[1]
    )


def _line_box(
    start: Tuple[float, float],
    end: Tuple[float, float],
    padding: float = 0.0,
) -> Tuple[float, float, float, float]:
    return (
        min(start[0], end[0]) - padding,
        min(start[1], end[1]) - padding,
        max(start[0], end[0]) + padding,
        max(start[1], end[1]) + padding,
    )


def _clip_anchor(
    anchor: Tuple[float, float],
    label_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    margin: int = 18,
) -> Tuple[float, float]:
    width, height = canvas_size
    label_w, label_h = label_size
    return (
        min(max(anchor[0], margin), max(margin, width - label_w - margin)),
        min(max(anchor[1], margin), max(margin, height - label_h - margin)),
    )


def _label_box_from_anchor(
    anchor: Tuple[float, float],
    label_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    padding_x = 10
    padding_y = 6
    return (
        anchor[0],
        anchor[1],
        anchor[0] + label_size[0] + padding_x * 2,
        anchor[1] + label_size[1] + padding_y * 2,
    )


def _point_to_box_distance(point: Tuple[float, float], box: Tuple[float, float, float, float]) -> float:
    dx = 0.0
    if point[0] < box[0]:
        dx = box[0] - point[0]
    elif point[0] > box[2]:
        dx = point[0] - box[2]

    dy = 0.0
    if point[1] < box[1]:
        dy = box[1] - point[1]
    elif point[1] > box[3]:
        dy = point[1] - box[3]

    return math.hypot(dx, dy)


def _closest_point_on_box(
    point: Tuple[float, float],
    box: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    return (
        min(max(point[0], box[0]), box[2]),
        min(max(point[1], box[1]), box[3]),
    )


def _pick_label_anchor(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    candidate_anchors: List[Tuple[float, float]],
    avoid_boxes: List[Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    text_w, text_h = _text_size(draw, text, font)
    padding_x = 10
    padding_y = 6

    for anchor in candidate_anchors:
        label_box = (
            anchor[0],
            anchor[1],
            anchor[0] + text_w + padding_x * 2,
            anchor[1] + text_h + padding_y * 2,
        )
        if not any(_boxes_overlap(label_box, avoid_box, padding=10.0) for avoid_box in avoid_boxes):
            return anchor

    return candidate_anchors[0]


def _wrap_words(text: str, max_chars: int, max_lines: int) -> Optional[str]:
    words = [word for word in str(text or "").split() if word]
    if not words:
        return str(text or "")

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines:
            return None
    lines.append(current)
    if len(lines) > max_lines:
        return None
    return "\n".join(lines)


def _truncate_text(text: str, max_chars: int, max_lines: int) -> str:
    compact = " ".join(str(text or "").split())
    max_total = max_chars * max_lines - 1
    if len(compact) <= max_total:
        wrapped = _wrap_words(compact, max_chars=max_chars, max_lines=max_lines)
        return wrapped if wrapped is not None else compact
    trimmed = compact[:max_total].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    trimmed = (trimmed or compact[: max_total - 1]).rstrip()
    return _wrap_words(f"{trimmed}…", max_chars=max_chars, max_lines=max_lines) or f"{trimmed}…"


def _build_text_variants(text: str, is_primary: bool) -> List[str]:
    compact = " ".join(str(text or "").split())
    if is_primary:
        return [compact]

    variants: List[str] = []
    full_wrapped = _wrap_words(compact, max_chars=18, max_lines=2)
    if full_wrapped is not None:
        variants.append(full_wrapped)
    variants.append(compact)
    variants.append(_truncate_text(compact, max_chars=16, max_lines=2))

    deduped: List[str] = []
    for variant in variants:
        if variant and variant not in deduped:
            deduped.append(variant)
    return deduped


def _generate_box_label_candidates(
    box: Tuple[float, float, float, float],
    is_primary: bool,
) -> List[Tuple[float, float]]:
    min_x, min_y, max_x, max_y = box
    if is_primary:
        return [
            (max_x + 14, min_y - 8),
            (min_x - 170, min_y - 12),
            (max_x + 14, max_y + 10),
            (min_x - 120, max_y + 10),
            (max_x + 28, (min_y + max_y) / 2.0 - 14),
            (min_x - 185, (min_y + max_y) / 2.0 - 14),
        ]
    return [
        (max_x + 10, min_y - 8),
        (min_x - 120, min_y - 8),
        (max_x + 10, max_y + 8),
        (min_x - 88, max_y + 8),
        ((min_x + max_x) / 2.0 - 36, min_y - 38),
        ((min_x + max_x) / 2.0 - 36, max_y + 10),
    ]


def _generate_radial_candidates(
    anchor_point: Tuple[float, float],
    preferred_offset: Tuple[float, float],
) -> List[Tuple[float, float]]:
    offsets = [
        preferred_offset,
        (preferred_offset[0], preferred_offset[1] - 28),
        (preferred_offset[0], preferred_offset[1] + 26),
        (preferred_offset[0] + 26, preferred_offset[1]),
        (preferred_offset[0] - 26, preferred_offset[1]),
        (preferred_offset[0] + 18, preferred_offset[1] - 18),
        (preferred_offset[0] - 18, preferred_offset[1] + 18),
    ]
    return [(anchor_point[0] + dx, anchor_point[1] + dy) for dx, dy in offsets]


def _placement_penalty(
    candidate_box: Tuple[float, float, float, float],
    avoid_boxes: List[Tuple[float, float, float, float]],
    anchor_point: Tuple[float, float],
    candidate_anchor: Tuple[float, float],
) -> float:
    penalty = 0.0
    for avoid_box in avoid_boxes:
        if _boxes_overlap(candidate_box, avoid_box, padding=8.0):
            overlap_w = min(candidate_box[2], avoid_box[2]) - max(candidate_box[0], avoid_box[0])
            overlap_h = min(candidate_box[3], avoid_box[3]) - max(candidate_box[1], avoid_box[1])
            penalty += max(1.0, overlap_w) * max(1.0, overlap_h) + 5000.0
    penalty += math.hypot(candidate_anchor[0] - anchor_point[0], candidate_anchor[1] - anchor_point[1]) * 0.6
    return penalty


def _place_label_item(
    draw: ImageDraw.ImageDraw,
    item: Dict[str, Any],
    canvas_size: Tuple[int, int],
    occupied_boxes: List[Tuple[float, float, float, float]],
) -> Dict[str, Any]:
    placed_item = dict(item)
    best: Optional[Tuple[float, str, Tuple[float, float], Tuple[float, float, float, float]]] = None

    for text_variant in item["text_options"]:
        label_size = _text_size(draw, text_variant, item["font"])
        for candidate_anchor in item["candidate_anchors"]:
            clipped_anchor = _clip_anchor(candidate_anchor, label_size, canvas_size)
            candidate_box = _label_box_from_anchor(clipped_anchor, label_size)
            penalty = _placement_penalty(
                candidate_box,
                occupied_boxes + item.get("avoid_boxes", []),
                item["anchor_point"],
                clipped_anchor,
            )
            if best is None or penalty < best[0]:
                best = (penalty, text_variant, clipped_anchor, candidate_box)
            if penalty < 1.0:
                placed_item["display_text"] = text_variant
                placed_item["placed_anchor"] = clipped_anchor
                placed_item["placed_bbox"] = candidate_box
                placed_item["leader_line"] = item.get("leader_line", False)
                return placed_item

    if best is None:
        placed_item["display_text"] = item["text_options"][0]
        placed_item["placed_anchor"] = item["candidate_anchors"][0]
        label_size = _text_size(draw, placed_item["display_text"], item["font"])
        placed_item["placed_bbox"] = _label_box_from_anchor(placed_item["placed_anchor"], label_size)
        placed_item["leader_line"] = item.get("leader_line", False)
        return placed_item

    _, best_text, best_anchor, best_box = best
    placed_item["display_text"] = best_text
    placed_item["placed_anchor"] = best_anchor
    placed_item["placed_bbox"] = best_box
    placed_item["leader_line"] = item.get("leader_line", False)
    return placed_item


def _draw_leader_line(
    draw: ImageDraw.ImageDraw,
    target: Tuple[float, float],
    label_box: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
) -> None:
    end_point = _closest_point_on_box(target, label_box)
    draw.line((target[0], target[1], end_point[0], end_point[1]), fill=color, width=2)


def _build_box_corners(center: Tuple[float, float, float], half_scale: float, height: float) -> List[Tuple[float, float, float]]:
    cx, cy, cz = center
    return [
        (cx - half_scale, cy, cz - half_scale),
        (cx + half_scale, cy, cz - half_scale),
        (cx + half_scale, cy, cz + half_scale),
        (cx - half_scale, cy, cz + half_scale),
        (cx - half_scale, cy + height, cz - half_scale),
        (cx + half_scale, cy + height, cz - half_scale),
        (cx + half_scale, cy + height, cz + half_scale),
        (cx - half_scale, cy + height, cz + half_scale),
    ]


def _draw_box(
    draw: ImageDraw.ImageDraw,
    camera: Dict[str, Any],
    center: Tuple[float, float, float],
    scale: float,
    height: float,
    outline: Tuple[int, int, int],
    face_fill: Optional[Tuple[int, int, int, int]],
    line_width: int,
) -> Optional[Tuple[float, float, float, float]]:
    corners_3d = _build_box_corners(center, scale * 0.5, height)
    projected = [_project_point(point, camera) for point in corners_3d]
    if any(point is None for point in projected):
        return None

    points_2d = [(point[0], point[1]) for point in projected if point is not None]
    top_face = [points_2d[index] for index in (4, 5, 6, 7)]
    front_face = [points_2d[index] for index in (0, 1, 5, 4)]
    side_face = [points_2d[index] for index in (1, 2, 6, 5)]

    if face_fill is not None:
        draw.polygon(side_face, fill=face_fill)
        draw.polygon(front_face, fill=face_fill)
        draw.polygon(top_face, fill=(255, 255, 255, min(200, face_fill[3] + 36)))

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start_index, end_index in edges:
        start = projected[start_index]
        end = projected[end_index]
        if start is None or end is None:
            continue
        draw.line((start[0], start[1], end[0], end[1]), fill=outline, width=line_width)

    box_bounds = (
        min(point[0] for point in points_2d),
        min(point[1] for point in points_2d),
        max(point[0] for point in points_2d),
        max(point[1] for point in points_2d),
    )
    return box_bounds


def _draw_arrow_2d(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_len = max(10.0, width * 2.2)
    left = (
        end[0] - arrow_len * math.cos(angle - math.pi / 6.0),
        end[1] - arrow_len * math.sin(angle - math.pi / 6.0),
    )
    right = (
        end[0] - arrow_len * math.cos(angle + math.pi / 6.0),
        end[1] - arrow_len * math.sin(angle + math.pi / 6.0),
    )
    draw.polygon([end, left, right], fill=color)


def _draw_main_object_axes(
    draw: ImageDraw.ImageDraw,
    camera: Dict[str, Any],
    position: List[int],
    facing: Optional[str],
    base_height: float,
    axis_length: float,
    font: ImageFont.ImageFont,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float, float, float]]]:
    origin = _grid_to_world(position)
    origin = (origin[0], base_height * 0.55, origin[2])

    up_axis = (0.0, 1.0, 0.0)
    front_axis = _direction_to_ground_vector(facing, mode="object")
    if abs(front_axis[1]) > 0.9:
        front_axis = (0.0, 0.0, 1.0)
    left_axis = _cross(up_axis, front_axis)
    if _norm(left_axis) < 1e-6:
        left_axis = (-1.0, 0.0, 0.0)
    left_axis = _normalize(left_axis)
    front_axis = _normalize(front_axis)

    axis_specs = [
        ("Front", front_axis, (214, 51, 48), (8, -4)),
        ("Left", left_axis, (91, 148, 74), (-48, -22)),
        ("Up", up_axis, (58, 93, 196), (-12, -34)),
    ]

    origin_2d = _project_point(origin, camera)
    if origin_2d is None:
        return [], []

    label_items: List[Dict[str, Any]] = []
    label_boxes: List[Tuple[float, float, float, float]] = []
    for label, axis_vec, color, label_offset in axis_specs:
        end_3d = _vec_add(origin, _vec_mul(axis_vec, axis_length))
        end_2d = _project_point(end_3d, camera)
        if end_2d is None:
            continue
        _draw_arrow_2d(draw, (origin_2d[0], origin_2d[1]), (end_2d[0], end_2d[1]), color, width=4)
        line_guard = _expand_box(
            _line_box((origin_2d[0], origin_2d[1]), (end_2d[0], end_2d[1]), padding=5.0),
            8.0,
        )
        tip_guard = _expand_box(
            (end_2d[0] - 10, end_2d[1] - 10, end_2d[0] + 10, end_2d[1] + 10),
            4.0,
        )
        label_items.append(
            {
                "label_type": "axis",
                "priority": 100,
                "text_options": [label],
                "font": font,
                "fill": color,
                "background": (255, 255, 255, 192),
                "anchor_point": (end_2d[0], end_2d[1]),
                "candidate_anchors": _generate_radial_candidates((end_2d[0], end_2d[1]), label_offset),
                "avoid_boxes": [line_guard, tip_guard],
                "leader_line": False,
            }
        )
        label_boxes.extend([line_guard, tip_guard])
    return label_items, label_boxes


def _draw_view_marker(
    draw: ImageDraw.ImageDraw,
    camera: Dict[str, Any],
    position: List[int],
    facing: Optional[str],
    label: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    label_offset: Tuple[float, float],
) -> Optional[Tuple[Dict[str, Any], List[Tuple[float, float, float, float]]]]:
    origin = _grid_to_world(position)
    start_3d = (origin[0], 0.12, origin[2])
    end_3d = _vec_add(start_3d, _vec_mul(_direction_to_ground_vector(facing, mode="view"), 1.3))

    start_2d = _project_point(start_3d, camera)
    end_2d = _project_point(end_3d, camera)
    if start_2d is None or end_2d is None:
        return None

    radius = 7
    ellipse_bbox = (
        start_2d[0] - radius,
        start_2d[1] - radius,
        start_2d[0] + radius,
        start_2d[1] + radius,
    )
    try:
        draw.ellipse(
            ellipse_bbox,
            fill=color + (255,),
            outline=(255, 255, 255, 220),
            width=2,
        )
    except TypeError:
        draw.ellipse(
            ellipse_bbox,
            fill=color + (255,),
            outline=(255, 255, 255, 220),
        )
    _draw_arrow_2d(draw, (start_2d[0], start_2d[1]), (end_2d[0], end_2d[1]), color, width=4)
    line_guard = _expand_box(
        _line_box((start_2d[0], start_2d[1]), (end_2d[0], end_2d[1]), padding=5.0),
        8.0,
    )
    marker_guard = _expand_box(ellipse_bbox, 5.0)
    return (
        {
            "label_type": "view",
            "priority": 80,
            "text_options": [f"{label}: {facing or 'none'}"],
            "font": font,
            "fill": color,
            "background": (255, 255, 255, 180),
            "anchor_point": (end_2d[0], end_2d[1]),
            "candidate_anchors": _generate_radial_candidates((end_2d[0], end_2d[1]), label_offset),
            "avoid_boxes": [line_guard, marker_guard],
            "leader_line": False,
        },
        [line_guard, marker_guard],
    )


def _draw_arrow(draw: ImageDraw.ImageDraw, center: Tuple[float, float], direction: Optional[str], length: float, color: Tuple[int, int, int], width: int = 5) -> Optional[Tuple[float, float]]:
    if not direction or direction not in _DIR_TO_VEC:
        return None

    dx, dy = _DIR_TO_VEC[direction]
    end_x = center[0] + dx * length
    end_y = center[1] + dy * length
    draw.line((center[0], center[1], end_x, end_y), fill=color, width=width)

    arrow_size = max(8.0, length * 0.18)
    if dx == 0 and dy == -1:
        points = [(end_x, end_y), (end_x - arrow_size * 0.55, end_y + arrow_size), (end_x + arrow_size * 0.55, end_y + arrow_size)]
    elif dx == 0 and dy == 1:
        points = [(end_x, end_y), (end_x - arrow_size * 0.55, end_y - arrow_size), (end_x + arrow_size * 0.55, end_y - arrow_size)]
    elif dx == -1 and dy == 0:
        points = [(end_x, end_y), (end_x + arrow_size, end_y - arrow_size * 0.55), (end_x + arrow_size, end_y + arrow_size * 0.55)]
    else:
        points = [(end_x, end_y), (end_x - arrow_size, end_y - arrow_size * 0.55), (end_x - arrow_size, end_y + arrow_size * 0.55)]
    draw.polygon(points, fill=color)
    return end_x, end_y


def _draw_rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    radius: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int = 1,
) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(bbox, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(bbox, fill=fill, outline=outline)


def render_scene_layout_image(layout: Optional[Dict[str, Any]], width: int = 960, height: int = 600) -> Optional[Image.Image]:
    if not layout:
        return None

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    camera = _make_camera(width, height)
    title_font = _get_font(28)
    label_font = _get_font(18)
    small_font = _get_font(16)

    _draw_gradient_background(draw, width, height)
    _draw_ground_plane(draw, camera, width, height)

    scene_type = layout.get("scene_type", "unknown")
    orientation_mode = layout.get("orientation_mode", "object")
    title = f"Scene Overview: {scene_type}"
    subtitle = "3D object-facing scene" if orientation_mode == "object" else "3D view-facing scene"
    draw.text((40, 24), title, font=title_font, fill=(34, 46, 60, 255))
    draw.text((40, 58), subtitle, font=label_font, fill=(96, 103, 113, 255))

    main_object = layout.get("main_object")
    objects = layout.get("objects", [])
    views = layout.get("views", [])
    canvas_size = (width, height)
    label_items: List[Dict[str, Any]] = []
    occupied_boxes: List[Tuple[float, float, float, float]] = [
        (0.0, 0.0, float(width), 96.0),
        (0.0, float(height - 48), float(width), float(height)),
    ]
    sorted_objects = sorted(
        objects,
        key=lambda obj: _grid_to_world(obj["position"])[2],
        reverse=True,
    )

    for obj in sorted_objects:
        is_main = obj.get("render_role") == "main" or obj.get("name") == main_object
        world_center = _grid_to_world(obj["position"])
        scale = float(obj.get("render_scale", 0.8))
        box_height = float(obj.get("render_height", 1.1))
        outline = (167, 134, 52) if is_main else (88, 125, 190)
        fill = (220, 192, 106, 68) if is_main else (118, 156, 218, 40)
        box_bounds = _draw_box(
            draw,
            camera,
            world_center,
            scale,
            box_height,
            outline=outline,
            face_fill=fill,
            line_width=4 if is_main else 3,
        )
        if box_bounds is None:
            continue

        occupied_boxes.append(_expand_box(box_bounds, 8.0))
        avoid_boxes: List[Tuple[float, float, float, float]] = []
        if is_main and scene_type == "among":
            axis_items, axis_boxes = _draw_main_object_axes(
                draw,
                camera,
                obj["position"],
                obj.get("facing") or layout.get("main_object_facing"),
                box_height,
                axis_length=max(1.5, box_height * 1.15),
                font=label_font,
            )
            label_items.extend(axis_items)
            avoid_boxes.extend(axis_boxes)
            occupied_boxes.extend(axis_boxes)

        label_items.append(
            {
                "label_type": "object_main" if is_main else "object_secondary",
                "priority": 90 if is_main else 40,
                "text_options": _build_text_variants(obj["name"], is_primary=is_main),
                "font": label_font,
                "fill": (24, 32, 42) if is_main else (36, 52, 76),
                "background": (255, 255, 255, 220) if is_main else (255, 255, 255, 188),
                "anchor_point": (
                    (box_bounds[0] + box_bounds[2]) / 2.0,
                    box_bounds[1] if is_main else (box_bounds[1] + box_bounds[3]) / 2.0,
                ),
                "candidate_anchors": _generate_box_label_candidates(box_bounds, is_primary=is_main),
                "avoid_boxes": avoid_boxes,
                "leader_line": not is_main,
                "leader_target": (
                    (box_bounds[0] + box_bounds[2]) / 2.0,
                    box_bounds[1] if is_main else (box_bounds[1] + box_bounds[3]) / 2.0,
                ),
            }
        )

    view_palette = [
        (45, 117, 84),
        (38, 141, 115),
        (60, 102, 176),
        (154, 86, 38),
        (124, 72, 168),
        (184, 82, 82),
    ]
    label_offsets = [
        (10, -30),
        (10, 10),
        (-84, -30),
        (-84, 10),
        (18, -52),
        (-96, 26),
    ]
    for index, view in enumerate(views):
        if scene_type == "among":
            continue
        marker_result = _draw_view_marker(
            draw,
            camera,
            view["position"],
            view.get("facing"),
            view["name"],
            view_palette[index % len(view_palette)],
            small_font,
            label_offsets[index % len(label_offsets)],
        )
        if marker_result is None:
            continue
        view_label_item, view_reserved_boxes = marker_result
        label_items.append(view_label_item)
        occupied_boxes.extend(view_reserved_boxes)

    placed_label_items: List[Dict[str, Any]] = []
    for item in sorted(label_items, key=lambda current: current["priority"], reverse=True):
        placed_item = _place_label_item(draw, item, canvas_size, occupied_boxes)
        placed_label_items.append(placed_item)
        occupied_boxes.append(_expand_box(placed_item["placed_bbox"], 6.0))

    for item in sorted(placed_label_items, key=lambda current: current["priority"]):
        bbox = _draw_label(
            draw,
            item["display_text"],
            item["placed_anchor"],
            item["font"],
            item["fill"],
            item["background"],
        )
        if item.get("leader_line") and item.get("leader_target") is not None:
            if _point_to_box_distance(item["leader_target"], bbox) > 16.0:
                _draw_leader_line(draw, item["leader_target"], bbox, item["fill"])

    footer = "Main object uses explicit 3D axes; around/rotation emphasize camera heading."
    draw.text((40, height - 34), footer, font=small_font, fill=(88, 96, 104, 220))
    return image.convert("RGB")


def save_scene_layout(output_dir: str, layout: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    if not output_dir or not layout:
        return {
            "scene_layout_path": None,
            "scene_orientation_path": None,
            "main_object_orientation_path": None,
            "scene_overview_path": None,
        }

    os.makedirs(output_dir, exist_ok=True)
    scene_layout_path = os.path.join(output_dir, "scene_objects.json")
    with open(scene_layout_path, "w", encoding="utf-8") as handle:
        json.dump(layout, handle, ensure_ascii=False, indent=2)

    orientation_payload = {
        "scene_id": layout.get("scene_id"),
        "scene_type": layout.get("scene_type"),
        "orientation_mode": layout.get("orientation_mode"),
        "main_object": layout.get("main_object"),
        "main_object_facing": layout.get("main_object_facing"),
        "views": layout.get("views", []),
    }
    scene_orientation_path = os.path.join(output_dir, "scene_orientation.json")
    with open(scene_orientation_path, "w", encoding="utf-8") as handle:
        json.dump(orientation_payload, handle, ensure_ascii=False, indent=2)

    main_object_orientation_path = None
    if layout.get("main_object") and layout.get("main_object_facing"):
        main_object_orientation_path = os.path.join(output_dir, "main_object_orientation.json")
        with open(main_object_orientation_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "scene_id": layout.get("scene_id"),
                    "name": layout.get("main_object"),
                    "facing": layout.get("main_object_facing"),
                    "orientation_mode": layout.get("orientation_mode"),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    scene_overview_path = os.path.join(output_dir, "scene_overview.png")
    image = render_scene_layout_image(layout)
    if image is not None:
        image.save(scene_overview_path)
    else:
        scene_overview_path = None

    return {
        "scene_layout_path": scene_layout_path,
        "scene_orientation_path": scene_orientation_path,
        "main_object_orientation_path": main_object_orientation_path,
        "scene_overview_path": scene_overview_path,
    }


def load_scene_layout(scene_dir: str) -> Optional[Dict[str, Any]]:
    scene_layout_path = os.path.join(scene_dir, "scene_objects.json")
    if not os.path.exists(scene_layout_path):
        return None
    with open(scene_layout_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_scene_overview_image(
    scene_id: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    processed_base_dir: Optional[str] = None,
    width: int = 960,
    height: int = 600,
) -> Optional[Image.Image]:
    if result is not None:
        maybe_image = result.get("_scene_overview_image")
        if isinstance(maybe_image, Image.Image):
            return maybe_image.copy()

        overview_path = result.get("_scene_overview_path")
        if overview_path:
            layout = load_scene_layout(os.path.dirname(overview_path))
            if layout:
                return render_scene_layout_image(layout, width=width, height=height)
            if os.path.exists(overview_path):
                return Image.open(overview_path).convert("RGB")

    if processed_base_dir and scene_id:
        scene_dir = os.path.join(processed_base_dir, scene_id)
        layout = load_scene_layout(scene_dir)
        if layout:
            return render_scene_layout_image(layout, width=width, height=height)
        overview_path = os.path.join(scene_dir, "scene_overview.png")
        if os.path.exists(overview_path):
            return Image.open(overview_path).convert("RGB")

    if result:
        layout = build_scene_layout(result)
        if layout:
            return render_scene_layout_image(layout, width=width, height=height)

    return None
