"""3D AABB debug visualization for object extraction results."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


FLOOR_KEYWORDS = ("floor", "carpet", "rug")
FLOATING_KEYWORDS = ("wall", "ceiling", "fireplace", "window", "picture", "tv", "television")
OBJECT_COLORS = [
    "#3B6FB6", "#D05A40", "#4E9A51", "#8A62C4", "#D08B2D",
    "#2F9AA0", "#C34D88", "#7A8A2C", "#5C6BC0", "#B25D32",
]
# Display-only unit scaling. These values only affect the debug plot and never
# modify extracted 3D coordinates or JSON output. X/forward are expanded to make
# nearby objects easier to distinguish; height is kept slightly compressed.
PLOT_X_SCALE = 2.2
PLOT_FORWARD_SCALE = 2.2
PLOT_HEIGHT_SCALE = 1.15
MIN_FOOTPRINT_DISPLAY_SIZE = 0.16
OBJECT_DEBUG_LABEL_WIDTH = 190
OBJECT_DEBUG_THUMB_GAP = 12
OBJECT_DEBUG_ROW_GAP = 10
OBJECT_DEBUG_HEADER_HEIGHT = 26
OBJECT_DEBUG_MIN_ROW_HEIGHT = 100
FLOOR_DISPLAY_OFFSET = 0.025
FLOOR_DISPLAY_THICKNESS = 0.025


# The extracted 3D points are in the current camera coordinate system:
# x: image horizontal direction, positive to the right
# y: image vertical direction after sign flip, positive upward in camera coordinates
# z: negative depth, forward from the camera
#
# The floor plane drawn here is only a camera-coordinate visual reference.
# It is not a calibrated real-world ground plane.


def render_3d_aabb_scene(
    results: dict,
    output_path: str,
    width: int = 1200,
    height: int = 700,
    show_floor: bool = True,
    show_camera: bool = True,
    view: str = "top",
) -> str:
    """Render extracted object 3D AABBs as an oblique top-down 3D PNG."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    objects = _valid_3d_objects(results)
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F6F2")
    ax.set_title(
        "3D AABB Top-Down View\nplot axes: X=image right, Y=forward(-Z), Z=height(camera Y)",
        pad=16,
        color="#1F2933",
    )
    ax.set_xlabel("X: image right", labelpad=8)
    ax.set_ylabel("Forward: -Z depth", labelpad=8)
    ax.set_zlabel("Height: Y camera", labelpad=8)
    _style_3d_axes(ax)

    if not objects:
        _draw_empty_3d_scene(ax, show_camera)
        _set_3d_view(ax, view)
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return str(output)

    floor_y = _estimate_floor_y(objects)
    floor_bounds = _floor_plan_bounds(objects)
    plot_points = []
    if show_floor:
        _draw_3d_floor_grid(ax, floor_y, floor_bounds)
        plot_points.extend(_floor_plot_points(floor_y, floor_bounds))

    draw_objects = _sort_3d_objects_for_drawing(objects)
    legend_handles = []
    for idx, obj in enumerate(draw_objects):
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        corners = _plot_box_corners(obj)
        plot_points.extend(corners)
        _draw_3d_box(ax, corners, color)
        center = _plot_point(obj["position"])
        plot_points.append(center)
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=42, edgecolors="white", linewidths=0.8, depthshade=False)
        _draw_3d_number_label(ax, obj, center, idx + 1, color)
        legend_handles.append(Line2D([0], [0], color=color, lw=3, label=f"{idx + 1}. {_legend_label(obj, floor_y)}"))

    if show_camera:
        _draw_3d_camera(ax, floor_y, floor_bounds)
        plot_points.extend(_camera_plot_points(floor_y, floor_bounds))

    _apply_3d_limits(ax, plot_points)
    _set_3d_view(ax, view)
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=7, frameon=True, facecolor="white", edgecolor="#D1D5DB")
    try:
        ax.dist = 7
    except Exception:
        pass
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.88)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def compose_question_image_3d_visualization(
    image_path_or_pil,
    question: str,
    aabb_vis_path: str,
    output_path: str,
    answer: str = "",
    results: Optional[dict] = None,
    debug_dir: Optional[str] = None,
    canvas_width: int = 1400,
) -> str:
    """Compose question text, input image, object debug crops, and the 3D AABB render."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_image = _load_image(image_path_or_pil)
    object_debug_size = _object_debug_thumb_size(source_image.size)
    canvas_width = max(canvas_width, _min_canvas_width_for_object_debug(object_debug_size))

    top_height = 600
    padding = 30
    gap = 20
    object_rows = _count_object_debug_rows(results or {}, debug_dir)
    object_row_height = _object_debug_row_height(object_debug_size)
    object_height = max(180, 42 + object_rows * object_row_height + max(0, object_rows - 1) * OBJECT_DEBUG_ROW_GAP)
    bottom_height = 850
    canvas_height = top_height + object_height + bottom_height + padding * 4
    left_width = int(canvas_width * 0.42)
    right_width = canvas_width - left_width - padding * 2 - gap
    bottom_width = canvas_width - padding * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(26, bold=True)
    body_font = _load_font(22)
    small_font = _load_font(16)

    question_box = (padding, padding, left_width, top_height)
    image_box = (padding + left_width + gap, padding, right_width, top_height)
    object_box = (padding, padding + top_height + gap, bottom_width, object_height)
    aabb_box = (padding, object_box[1] + object_height + gap, bottom_width, bottom_height)

    _draw_section_title(draw, "Question", question_box[0], question_box[1], title_font)
    question_text = question or ""
    if answer:
        question_text = f"{question_text}\n\nAnswer\n{answer}"
    spatial_info = _format_spatial_info_for_panel(results or {})
    if spatial_info:
        question_text = f"{question_text}\n\n{spatial_info}"
    wrapped = _wrap_and_truncate_text(question_text, body_font, question_box[2] - 12, question_box[3] - 58)
    draw.multiline_text((question_box[0], question_box[1] + 42), wrapped, fill=(30, 30, 30), font=body_font, spacing=6)

    _draw_section_title(draw, "Input Image", image_box[0], image_box[1], title_font)
    image = _fit_image(source_image, image_box[2], image_box[3] - 42)
    canvas.paste(image, (image_box[0] + (image_box[2] - image.width) // 2, image_box[1] + 42 + (image_box[3] - 42 - image.height) // 2))

    _draw_section_title(draw, "Object Debug Images", object_box[0], object_box[1], title_font)
    _draw_object_debug_images(canvas, draw, results or {}, debug_dir, object_box, small_font, object_debug_size)

    _draw_section_title(draw, "3D AABB Visualization", aabb_box[0], aabb_box[1], title_font)
    aabb_image = Image.open(aabb_vis_path).convert("RGB")
    aabb_image = _fit_image(aabb_image, aabb_box[2], aabb_box[3] - 42)
    canvas.paste(aabb_image, (aabb_box[0] + (aabb_box[2] - aabb_image.width) // 2, aabb_box[1] + 42 + (aabb_box[3] - 42 - aabb_image.height) // 2))

    canvas.save(output)
    return str(output)


def visualize_3d_debug_panel(
    image_path_or_pil,
    question: str,
    results: dict,
    save_dir: str,
    prefix: str = "3d_debug",
    answer: str = "",
) -> dict:
    """Create the workflow debug outputs: 3d_aabb.png and 3d_debug_panel.png."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    aabb_path = save_path / "3d_aabb.png"
    panel_path = save_path / f"{prefix}_panel.png"
    render_3d_aabb_scene(results, str(aabb_path))
    compose_question_image_3d_visualization(
        image_path_or_pil,
        question,
        str(aabb_path),
        str(panel_path),
        answer=answer,
        results=results,
        debug_dir=str(save_path),
    )
    return {"aabb_vis_path": str(aabb_path), "panel_path": str(panel_path)}


def _format_spatial_info_for_panel(results: dict, max_lines: int = 8) -> str:
    lines = ["Extracted 3D Objects"]
    omitted = 0
    for name, item in (results or {}).items():
        if len(lines) >= max_lines:
            omitted += 1
            continue
        if not isinstance(item, dict):
            continue
        if item.get("error"):
            lines.append(f"{name}: error={item.get('error')}")
            continue
        position = _format_vector_for_panel(item.get("position"))
        center = _format_vector_for_panel(item.get("box3d_center"))
        size = _format_vector_for_panel(item.get("box3d_size"))
        if position is None or center is None or size is None:
            continue
        lines.append(f"{name}: pos={position}, center={center}, size={size}")
    if omitted:
        lines.append(f"... {omitted} more objects")
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_vector_for_panel(value: Any) -> Optional[str]:
    vector = _to_float_list(value, 3)
    if vector is None:
        return None
    return "[" + ", ".join(f"{item:.3f}" for item in vector) + "]"


def _valid_3d_objects(results: dict) -> List[Dict[str, Any]]:
    objects = []
    for name, item in (results or {}).items():
        if not isinstance(item, dict) or item.get("error"):
            continue
        min_pt = _to_float_list(item.get("box3d_min"), 3)
        max_pt = _to_float_list(item.get("box3d_max"), 3)
        position = _to_float_list(item.get("position"), 3)
        if min_pt is None or max_pt is None or position is None:
            continue
        objects.append({"name": str(name), "item": item, "box3d_min": min_pt, "box3d_max": max_pt, "position": position})
    return objects


def _sort_3d_objects_for_drawing(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    floor_like = [obj for obj in objects if _is_floor_like_object(obj["name"])]
    others = [obj for obj in objects if not _is_floor_like_object(obj["name"])]
    floor_like.sort(key=lambda obj: obj["name"].lower())
    others.sort(key=_object_forward_depth, reverse=True)
    return floor_like + others


def _is_floor_like_object(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in FLOOR_KEYWORDS)


def _object_forward_depth(obj: Dict[str, Any]) -> float:
    z_values = [obj["position"][2], obj["box3d_min"][2], obj["box3d_max"][2]]
    return float(-np.mean(z_values))


def _to_float_list(value: Any, length: int) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.size < length or not np.all(np.isfinite(array[:length])):
        return None
    return [float(v) for v in array[:length]]


def _estimate_floor_y(objects: List[Dict[str, Any]]) -> float:
    floor_like = [obj for obj in objects if any(keyword in obj["name"].lower() for keyword in FLOOR_KEYWORDS)]
    if floor_like:
        return float(min(obj["box3d_min"][1] for obj in floor_like))
    return float(np.percentile([obj["box3d_min"][1] for obj in objects], 10))


def _floor_plan_bounds(objects: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    all_x = [0.0]
    all_forward = [0.0]
    for obj in objects:
        x0, x1, y0, y1 = _object_footprint(obj)
        all_x.extend([x0, x1])
        all_forward.extend([y0, y1])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_forward), max(all_forward)
    margin = max(x_max - x_min, y_max - y_min) * 0.08
    if margin <= 1e-6:
        margin = 0.5
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


def _object_footprint(obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x0 = min(_plot_x(obj["box3d_min"][0]), _plot_x(obj["box3d_max"][0]))
    x1 = max(_plot_x(obj["box3d_min"][0]), _plot_x(obj["box3d_max"][0]))
    forward0 = min(_plot_forward(obj["box3d_min"][2]), _plot_forward(obj["box3d_max"][2]))
    forward1 = max(_plot_forward(obj["box3d_min"][2]), _plot_forward(obj["box3d_max"][2]))
    if abs(x1 - x0) < MIN_FOOTPRINT_DISPLAY_SIZE:
        center = (x0 + x1) / 2.0
        x0 = center - MIN_FOOTPRINT_DISPLAY_SIZE / 2.0
        x1 = center + MIN_FOOTPRINT_DISPLAY_SIZE / 2.0
    if abs(forward1 - forward0) < MIN_FOOTPRINT_DISPLAY_SIZE:
        center = (forward0 + forward1) / 2.0
        forward0 = center - MIN_FOOTPRINT_DISPLAY_SIZE / 2.0
        forward1 = center + MIN_FOOTPRINT_DISPLAY_SIZE / 2.0
    return float(x0), float(x1), float(forward0), float(forward1)


def _draw_3d_floor_grid(ax, floor_y: float, bounds: Tuple[float, float, float, float]) -> None:
    x_min, x_max, forward_min, forward_max = bounds
    plot_floor_y = _plot_height(floor_y)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(forward_min, forward_max, 2))
    zz = np.full_like(xx, plot_floor_y)
    ax.plot_surface(xx, yy, zz, color="#EAE7DF", alpha=0.35, shade=False, linewidth=0)
    grid_color = "#BDB7AA"
    for x in np.linspace(x_min, x_max, 13):
        ax.plot([x, x], [forward_min, forward_max], [plot_floor_y, plot_floor_y], color=grid_color, linewidth=0.8, alpha=0.72)
    for forward in np.linspace(forward_min, forward_max, 13):
        ax.plot([x_min, x_max], [forward, forward], [plot_floor_y, plot_floor_y], color=grid_color, linewidth=0.8, alpha=0.72)


def _floor_plot_points(floor_y: float, bounds: Tuple[float, float, float, float]) -> List[List[float]]:
    x_min, x_max, forward_min, forward_max = bounds
    plot_floor_y = _plot_height(floor_y)
    return [[x_min, forward_min, plot_floor_y], [x_max, forward_max, plot_floor_y], [x_min, forward_max, plot_floor_y], [x_max, forward_min, plot_floor_y]]


def _plot_x(x: float) -> float:
    return float(x) * PLOT_X_SCALE


def _plot_forward(z: float) -> float:
    return float(-z) * PLOT_FORWARD_SCALE


def _plot_height(y: float) -> float:
    return float(y) * PLOT_HEIGHT_SCALE


def _plot_object_bottom_height(obj: Dict[str, Any]) -> float:
    y = min(obj["box3d_min"][1], obj["box3d_max"][1])
    if _is_floor_like_object(obj["name"]):
        y -= FLOOR_DISPLAY_OFFSET + FLOOR_DISPLAY_THICKNESS
    return _plot_height(y)


def _plot_object_top_height(obj: Dict[str, Any]) -> float:
    y = max(obj["box3d_min"][1], obj["box3d_max"][1])
    if _is_floor_like_object(obj["name"]):
        y = min(obj["box3d_min"][1], obj["box3d_max"][1]) - FLOOR_DISPLAY_OFFSET
    return _plot_height(y)


def _plot_point(point: List[float]) -> List[float]:
    return [_plot_x(point[0]), _plot_forward(point[2]), _plot_height(point[1])]


def _plot_box_corners(obj: Dict[str, Any]) -> List[List[float]]:
    x0, x1, forward0, forward1 = _object_footprint(obj)
    y0 = _plot_object_bottom_height(obj)
    y1 = _plot_object_top_height(obj)
    return [
        [x0, forward0, y0], [x1, forward0, y0], [x1, forward0, y1], [x0, forward0, y1],
        [x0, forward1, y0], [x1, forward1, y0], [x1, forward1, y1], [x0, forward1, y1],
    ]


def _draw_3d_box(ax, corners: List[List[float]], color: str) -> None:
    faces = [
        [corners[i] for i in (0, 1, 2, 3)],
        [corners[i] for i in (4, 5, 6, 7)],
        [corners[i] for i in (0, 1, 5, 4)],
        [corners[i] for i in (1, 2, 6, 5)],
        [corners[i] for i in (2, 3, 7, 6)],
        [corners[i] for i in (3, 0, 4, 7)],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=1.3, alpha=0.20))
    for start, end in [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]:
        p0 = corners[start]
        p1 = corners[end]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=2.3, alpha=0.96)


def _draw_3d_number_label(ax, obj: Dict[str, Any], center: List[float], number: int, color: str) -> None:
    top_z = _plot_object_top_height(obj)
    bottom_z = _plot_object_bottom_height(obj)
    label_z = top_z + max(0.04, abs(top_z - bottom_z) * 0.12)
    ax.text(
        center[0], center[1], label_z, str(number),
        color="#1F2933", fontsize=9, ha="center", va="center",
        bbox={"boxstyle": "circle,pad=0.24", "facecolor": "white", "edgecolor": color, "alpha": 0.96, "linewidth": 1.2},
    )


def _legend_label(obj: Dict[str, Any], floor_y: float) -> str:
    y_size = max(0.0, obj["box3d_max"][1] - obj["box3d_min"][1])
    parts = [obj["name"], f"h={y_size:.2f}"]
    if _is_floating_object(obj["name"]) or obj["box3d_min"][1] > floor_y + max(0.08, y_size * 0.2):
        parts.append(f"floating y={obj['position'][1]:.2f}")
    mask = obj["item"].get("mask_used_for_3d")
    if mask:
        parts.append(f"mask={mask}")
    return " | ".join(parts)


def _is_floating_object(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in FLOATING_KEYWORDS)


def _draw_3d_camera(ax, floor_y: float, bounds: Tuple[float, float, float, float]) -> None:
    x_min, x_max, forward_min, forward_max = bounds
    span = max(x_max - x_min, forward_max - forward_min)
    body = max(span * 0.035, 0.05)
    front_len = max(span * 0.12, 0.22)
    z = _plot_height(floor_y) + body * 0.45
    origin = [0.0, 0.0, z]
    plane = [
        [-body * 0.8, front_len * 0.55, z - body * 0.35],
        [body * 0.8, front_len * 0.55, z - body * 0.35],
        [body * 0.8, front_len * 0.55, z + body * 0.35],
        [-body * 0.8, front_len * 0.55, z + body * 0.35],
    ]
    ax.add_collection3d(Poly3DCollection([plane], facecolors="#D9B44A", edgecolors="#8A6A1F", linewidths=1.2, alpha=0.30))
    for corner in plane:
        ax.plot([origin[0], corner[0]], [origin[1], corner[1]], [origin[2], corner[2]], color="#8A6A1F", linewidth=1.6)
    for start, end in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        p0 = plane[start]
        p1 = plane[end]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#8A6A1F", linewidth=1.6)
    ax.scatter([0.0], [0.0], [z], s=42, color="#8A6A1F", edgecolors="white", linewidths=0.8, depthshade=False)
    _draw_3d_axis_line(ax, [0, 0, z], [0, front_len, z], "Front (-Z)", "#D33F3F")
    _draw_3d_axis_line(ax, [0, 0, z], [front_len * 0.70, 0, z], "Right (+X)", "#4A9A44")
    _draw_3d_axis_line(ax, [0, 0, z], [0, 0, z + front_len * 0.45], "Y up", "#2F5CC0")
    ax.text(0.0, -body * 1.4, z, "camera", color="#1F2933", fontsize=8, ha="center", bbox={"facecolor": "white", "edgecolor": "#8A6A1F", "alpha": 0.9, "pad": 2})


def _camera_plot_points(floor_y: float, bounds: Tuple[float, float, float, float]) -> List[List[float]]:
    x_min, x_max, forward_min, forward_max = bounds
    span = max(x_max - x_min, forward_max - forward_min)
    front_len = max(span * 0.12, 0.22)
    plot_floor_y = _plot_height(floor_y)
    return [[0, 0, plot_floor_y], [0, front_len, plot_floor_y + front_len * 0.45], [front_len * 0.70, 0, plot_floor_y]]


def _draw_3d_axis_line(ax, start: List[float], end: List[float], label: str, color: str) -> None:
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=2.1)
    ax.text(end[0], end[1], end[2], label, color=color, fontsize=8, ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})


def _draw_empty_3d_scene(ax, show_camera: bool) -> None:
    bounds = (-1.0, 1.0, -0.3, 1.3)
    floor_y = 0.0
    _draw_3d_floor_grid(ax, floor_y, bounds)
    points = _floor_plot_points(floor_y, bounds)
    if show_camera:
        _draw_3d_camera(ax, floor_y, bounds)
        points.extend(_camera_plot_points(floor_y, bounds))
    _apply_3d_limits(ax, points)
    ax.text2D(0.42, 0.54, "No valid 3D boxes", transform=ax.transAxes, fontsize=16, color="#1F2933")


def _apply_3d_limits(ax, points: List[List[float]]) -> None:
    arr = np.asarray(points, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    xy_radius = max(float(max(maxs[0] - mins[0], maxs[1] - mins[1])) / 2.0, 0.5)
    xy_center = [(mins[0] + maxs[0]) / 2.0, (mins[1] + maxs[1]) / 2.0]
    ax.set_xlim(xy_center[0] - xy_radius, xy_center[0] + xy_radius)
    ax.set_ylim(xy_center[1] - xy_radius, xy_center[1] + xy_radius)
    z_min = mins[2]
    z_max = maxs[2]
    z_pad = max((z_max - z_min) * 0.18, xy_radius * 0.08, 0.06)
    ax.set_zlim(z_min - z_pad * 0.10, z_max + z_pad)


def _set_3d_view(ax, view: str) -> None:
    if view == "camera":
        ax.view_init(elev=34, azim=-66)
    elif view == "free":
        ax.view_init(elev=40, azim=-54)
    else:
        ax.view_init(elev=48, azim=-58)


def _style_3d_axes(ax) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
    ax.tick_params(colors="#6B7280", labelsize=7, pad=2)


def _count_object_debug_rows(results: dict, debug_dir: Optional[str]) -> int:
    if not debug_dir:
        return 0
    debug_path = Path(debug_dir)
    count = 0
    for name, item in (results or {}).items():
        if not isinstance(item, dict):
            continue
        safe_name = _safe_debug_name(str(name))
        if any((debug_path / f"{safe_name}{suffix}").exists() for suffix in ("_box.png", "_mask.png", "_overlay.png")):
            count += 1
    return count


def _draw_object_debug_images(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    results: dict,
    debug_dir: Optional[str],
    box: Tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    thumb_size: Tuple[int, int],
) -> None:
    if not debug_dir:
        return
    debug_path = Path(debug_dir)
    x, y, width, height = box
    content_y = y + 42
    row_height = _object_debug_row_height(thumb_size)
    label_width = OBJECT_DEBUG_LABEL_WIDTH
    thumb_gap = OBJECT_DEBUG_THUMB_GAP
    thumb_width, thumb_height = thumb_size
    columns = [("box", "_box.png"), ("sam", "_mask.png"), ("overlay", "_overlay.png")]

    row = 0
    for name, item in (results or {}).items():
        if not isinstance(item, dict):
            continue
        safe_name = _safe_debug_name(str(name))
        image_paths = [(label, debug_path / f"{safe_name}{suffix}") for label, suffix in columns]
        if not any(path.exists() for _, path in image_paths):
            continue
        row_top = content_y + row * (row_height + OBJECT_DEBUG_ROW_GAP)
        if row_top + row_height > y + height:
            break
        draw.text((x, row_top + 4), _truncate_line(str(name), font, label_width - 8), fill=(30, 30, 30), font=font)
        for col_idx, (label, image_path) in enumerate(image_paths):
            col_x = x + label_width + col_idx * (thumb_width + thumb_gap)
            draw.text((col_x, row_top + 4), label, fill=(80, 80, 80), font=font)
            if not image_path.exists():
                continue
            try:
                thumb = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            thumb = _fit_image_allow_upscale(thumb, thumb_width, thumb_height)
            paste_x = col_x + (thumb_width - thumb.width) // 2
            paste_y = row_top + OBJECT_DEBUG_HEADER_HEIGHT + (thumb_height - thumb.height) // 2
            canvas.paste(thumb, (paste_x, paste_y))
        row += 1


def _object_debug_thumb_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = image_size
    return (max(1, width // 4), max(1, height // 4))


def _object_debug_row_height(thumb_size: Tuple[int, int]) -> int:
    return max(OBJECT_DEBUG_MIN_ROW_HEIGHT, OBJECT_DEBUG_HEADER_HEIGHT + thumb_size[1])


def _min_canvas_width_for_object_debug(thumb_size: Tuple[int, int]) -> int:
    return 60 + OBJECT_DEBUG_LABEL_WIDTH + OBJECT_DEBUG_THUMB_GAP * 2 + 3 * thumb_size[0]


def _safe_debug_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _load_image(image_path_or_pil) -> Image.Image:
    if isinstance(image_path_or_pil, Image.Image):
        return image_path_or_pil.convert("RGB")
    return Image.open(image_path_or_pil).convert("RGB")


def _fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((max_width, max_height), Image.LANCZOS)
    return image


def _fit_image_allow_upscale(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), Image.LANCZOS)


def _draw_section_title(draw: ImageDraw.ImageDraw, title: str, x: int, y: int, font: ImageFont.ImageFont) -> None:
    draw.text((x, y), title, fill=(0, 0, 0), font=font)


def _wrap_and_truncate_text(text: str, font: ImageFont.ImageFont, max_width: int, max_height: int) -> str:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(_wrap_line(paragraph, font, max_width))
    line_height = _text_bbox(font, "Ag")[3] + 6
    max_lines = max(1, max_height // max(1, line_height))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = _truncate_line(lines[-1], font, max_width)
    return "\n".join(lines)


def _wrap_line(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return [""]
    if " " not in text:
        return textwrap.wrap(text, width=max(8, max_width // 18)) or [text]
    lines = []
    current = ""
    for word in text.split():
        candidate = word if not current else current + " " + word
        if _text_bbox(font, candidate)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _truncate_line(line: str, font: ImageFont.ImageFont, max_width: int) -> str:
    suffix = "..."
    text = line
    while text and _text_bbox(font, text + suffix)[2] > max_width:
        text = text[:-1]
    return text + suffix if text else suffix


def _text_bbox(font: ImageFont.ImageFont, text: str) -> Tuple[int, int, int, int]:
    if hasattr(font, "getbbox"):
        return font.getbbox(text)
    width, height = font.getsize(text)
    return (0, 0, width, height)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()
