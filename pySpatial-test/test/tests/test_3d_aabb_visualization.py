import sys
from pathlib import Path

from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from object_3d_extraction.visualize_3d_aabb import (
    _format_spatial_info_for_panel,
    _object_debug_row_height,
    _object_debug_thumb_size,
    _plot_box_corners,
    _sort_3d_objects_for_drawing,
    _valid_3d_objects,
    render_3d_aabb_scene,
    visualize_3d_debug_panel,
)


def _image(tmpdir, size=(160, 100)):
    path = Path(str(tmpdir)) / "input.jpg"
    Image.new("RGB", size, color=(230, 235, 240)).save(path)
    return path


def _write_object_debug_images(tmpdir, object_name, size=(80, 60)):
    safe_name = object_name.replace("/", "_").replace(" ", "_")
    for suffix, color in [
        ("_box.png", (255, 0, 0)),
        ("_mask.png", (0, 255, 0)),
        ("_overlay.png", (0, 0, 255)),
    ]:
        Image.new("RGB", size, color=color).save(Path(str(tmpdir)) / f"{safe_name}{suffix}")


def _assert_outputs(paths):
    for value in paths.values():
        path = Path(value)
        assert path.exists()
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            width, height = image.size
        assert width > 100
        assert height > 100


def _base_results():
    return {
        "white coffee table": {
            "position": [0.1, -0.2, -1.5],
            "box3d_min": [-0.1, -0.3, -1.7],
            "box3d_max": [0.3, 0.0, -1.3],
            "box3d_size": [0.4, 0.3, 0.4],
            "box3d_center": [0.1, -0.15, -1.5],
            "depth_mode": 1.5,
            "num_points": 120,
        },
        "carpet": {
            "position": [0.0, -0.5, -2.0],
            "box3d_min": [-1.0, -0.55, -2.8],
            "box3d_max": [1.0, -0.45, -1.2],
            "box3d_size": [2.0, 0.1, 1.6],
            "box3d_center": [0.0, -0.5, -2.0],
            "depth_mode": 2.0,
            "num_points": 500,
        },
    }


def test_spatial_info_panel_text_includes_agent_3d_fields():
    text = _format_spatial_info_for_panel(_base_results())

    assert "Extracted 3D Objects" in text
    assert "white coffee table" in text
    assert "pos=[0.100, -0.200, -1.500]" in text
    assert "center=[0.100, -0.150, -1.500]" in text
    assert "size=[0.400, 0.300, 0.400]" in text


def test_visualization_with_fallback_fields(tmpdir):
    results = _base_results()
    results["white coffee table"].update({
        "mask_used_for_3d": "fallback",
        "mask_fallback_reason": "entity_box_too_large",
        "sam_mask_area": 8000,
        "final_mask_area": 2000,
    })
    results["carpet"].update({
        "mask_used_for_3d": "sam",
        "mask_fallback_reason": None,
        "sam_mask_area": 5000,
        "final_mask_area": 5000,
    })

    paths = visualize_3d_debug_panel(_image(tmpdir), "Where is the white coffee table?", results, str(tmpdir))

    _assert_outputs(paths)


def test_visualization_without_fallback_fields(tmpdir):
    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", _base_results(), str(tmpdir))

    _assert_outputs(paths)


def test_visualization_with_answer(tmpdir):
    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", _base_results(), str(tmpdir), answer="A. carpet")

    _assert_outputs(paths)


def test_visualization_with_object_debug_images(tmpdir):
    _write_object_debug_images(tmpdir, "white coffee table")
    _write_object_debug_images(tmpdir, "carpet")

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", _base_results(), str(tmpdir), answer="carpet")

    _assert_outputs(paths)
    with Image.open(paths["panel_path"]) as image:
        assert image.size[0] == 1400
        assert image.size[1] > 1260


def test_object_debug_images_use_half_input_image_size(tmpdir):
    input_size = (400, 240)
    expected_thumb_size = (100, 60)
    _write_object_debug_images(tmpdir, "white coffee table", size=input_size)
    _write_object_debug_images(tmpdir, "carpet", size=input_size)

    paths = visualize_3d_debug_panel(_image(tmpdir, size=input_size), "Question", _base_results(), str(tmpdir))

    _assert_outputs(paths)
    assert _object_debug_thumb_size(input_size) == expected_thumb_size
    with Image.open(paths["panel_path"]) as image:
        expected_object_height = 42 + 2 * _object_debug_row_height(expected_thumb_size) + 10
        assert image.size[1] == 600 + expected_object_height + 850 + 120


def test_visualization_with_partial_fallback_fields(tmpdir):
    results = _base_results()
    results["white coffee table"]["mask_used_for_3d"] = "fallback"
    results["white coffee table"]["mask_fallback_reason"] = "entity_box_too_large"

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", results, str(tmpdir))

    _assert_outputs(paths)


def test_visualization_with_none_fallback_reason(tmpdir):
    results = _base_results()
    results["white coffee table"]["mask_used_for_3d"] = "sam"
    results["white coffee table"]["mask_fallback_reason"] = None

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", results, str(tmpdir))

    _assert_outputs(paths)


def test_visualization_with_floor_like_object(tmpdir):
    results = _base_results()

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", results, str(tmpdir))

    _assert_outputs(paths)


def test_visualization_without_floor_like_object(tmpdir):
    results = {
        "chair": {
            "position": [0.2, -0.1, -1.0],
            "box3d_min": [0.1, -0.2, -1.2],
            "box3d_max": [0.3, 0.1, -0.8],
            "num_points": 50,
        }
    }

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", results, str(tmpdir))

    _assert_outputs(paths)


def test_visualization_with_error_object(tmpdir):
    results = _base_results()
    results["bad object"] = {"error": "No detection"}

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", results, str(tmpdir))

    _assert_outputs(paths)


def test_visualization_with_missing_3d_fields(tmpdir):
    results = {
        "missing box": {"position": [0.0, 0.0, -1.0], "box3d_min": [0.0, 0.0, -1.0]},
        "also missing": {"box3d_max": [1.0, 1.0, -2.0]},
    }

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", results, str(tmpdir))

    _assert_outputs(paths)



def test_render_3d_aabb_scene_supports_top_and_camera_views(tmpdir):
    results = _base_results()

    top_path = Path(str(tmpdir)) / "top.png"
    camera_path = Path(str(tmpdir)) / "camera.png"
    render_3d_aabb_scene(results, str(top_path), view="top")
    render_3d_aabb_scene(results, str(camera_path), view="camera")

    _assert_outputs({"top": str(top_path), "camera": str(camera_path)})


def test_visualization_with_suspended_object(tmpdir):
    results = _base_results()
    results["wall mounted tv"] = {
        "position": [0.6, 0.8, -1.6],
        "box3d_min": [0.4, 0.55, -1.8],
        "box3d_max": [0.8, 1.05, -1.4],
        "box3d_size": [0.4, 0.5, 0.4],
        "num_points": 80,
    }

    paths = visualize_3d_debug_panel(_image(tmpdir), "Question", results, str(tmpdir))

    _assert_outputs(paths)



def test_3d_objects_draw_floor_like_then_far_to_near():
    ordered = _sort_3d_objects_for_drawing(_valid_3d_objects(_base_results()))

    assert [obj["name"] for obj in ordered] == ["carpet", "white coffee table"]


def test_render_3d_aabb_scene_default_is_floor_plan(tmpdir):
    output = Path(str(tmpdir)) / "default.png"

    render_3d_aabb_scene(_base_results(), str(output))

    _assert_outputs({"default": str(output)})


def test_floor_like_box_is_displayed_below_table_box():
    objects = {obj["name"]: obj for obj in _valid_3d_objects(_base_results())}

    carpet_top = max(point[2] for point in _plot_box_corners(objects["carpet"]))
    table_bottom = min(point[2] for point in _plot_box_corners(objects["white coffee table"]))

    assert carpet_top < table_bottom
