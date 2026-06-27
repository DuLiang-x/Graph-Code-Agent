import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from mindcube import create_sample_flowchart, make_json_safe, summarize_result_for_output


def _box():
    return {
        "position": [0.0, 0.0, -3.0],
        "box3d_center": [0.0, 0.0, -3.0],
        "box3d_size": [1.0, 1.0, 1.0],
        "box3d_min": [-0.5, -0.5, -3.5],
        "box3d_max": [0.5, 0.5, -2.5],
        "orientation": [0.0, 0.0, -1.0],
    }


def test_create_sample_flowchart_appends_code_answer_and_correctness(tmpdir):
    image_path = Path(str(tmpdir)) / "image.png"
    Image.new("RGB", (80, 60), "white").save(str(image_path))
    result = {
        "scene_id": "omni3d_0",
        "question": "What is the ratio?",
        "images": [str(image_path)],
        "expected_answer": "1",
        "generated_answer": "1",
        "answer_reasoning": "Computed from 3D boxes.",
        "answer_correct": True,
        "fallback_used": False,
        "generated_code": "def program(input_scene):\n    return {'computed_results': {'answer': 1}}",
        "object_3d_boxes": {"tv": _box()},
    }

    output = create_sample_flowchart(result, Path(str(tmpdir)) / "flowcharts")

    assert output is not None
    assert Path(output).name == "omni3d_0_flowchart.png"
    assert Path(output).exists()


def test_make_json_safe_converts_numpy_values():
    payload = {"value": np.float32(1.5), "items": [np.array([1, 2])]}

    assert make_json_safe(payload) == {"value": 1.5, "items": [[1, 2]]}

def test_create_sample_flowchart_uses_extract_output_debug_dir(monkeypatch, tmpdir):
    image_path = Path(str(tmpdir)) / "image.png"
    Image.new("RGB", (80, 60), "white").save(str(image_path))
    extract_root = Path(str(tmpdir)) / "extract"
    sample_debug_dir = extract_root / "omni3d_0"
    sample_debug_dir.mkdir(parents=True)
    Image.new("RGB", (50, 40), "blue").save(str(sample_debug_dir / "3d_aabb.png"))
    Image.new("RGB", (20, 20), "red").save(str(sample_debug_dir / "tv_box.png"))
    calls = {}

    import object_3d_extraction.visualize_3d_aabb as viz

    def fake_compose(image_path_or_pil, question, aabb_vis_path, output_path, answer="", results=None, debug_dir=None, canvas_width=1400):
        calls["aabb_vis_path"] = aabb_vis_path
        calls["debug_dir"] = debug_dir
        Image.new("RGB", (100, 80), "white").save(output_path)
        return output_path

    monkeypatch.setattr(viz, "compose_question_image_3d_visualization", fake_compose)

    result = {
        "scene_id": "omni3d_0",
        "question": "Where is the tv?",
        "images": [str(image_path)],
        "expected_answer": "yes",
        "generated_answer": "yes",
        "question_type": "yes_no",
        "answer_correct": True,
        "fallback_used": False,
        "generated_code": "def program(input_scene): pass",
        "object_3d_boxes": {"tv": _box()},
        "mask_fallback": "auto",
    }

    output = create_sample_flowchart(result, Path(str(tmpdir)) / "flowcharts", extract_output_dir=str(extract_root))

    assert output is not None
    assert calls["debug_dir"] == str(sample_debug_dir)
    assert calls["aabb_vis_path"] == str(sample_debug_dir / "3d_aabb.png")


def test_summarize_result_for_output_removes_large_intermediate_fields():
    result = {
        "scene_id": "omni3d_0",
        "question": "Question",
        "question_type": "yes_no",
        "generated_answer": "yes",
        "generated_code": "def program(input_scene): pass",
        "answer_correct": True,
        "float_relative_error": 0.01,
        "float_mra": 1.0,
        "numeric_relative_error": 0.01,
        "numeric_mra": 1.0,
        "numeric_score": 1.0,
        "flowchart_path": "flowcharts/omni3d_0_flowchart.png",
        "object_3d_boxes": {"tv": _box()},
        "visual_clue": {"computed_results": {"answer": "yes"}},
        "generated_response": "large model response",
    }

    summary = summarize_result_for_output(result)

    assert summary["scene_id"] == "omni3d_0"
    assert summary["generated_answer"] == "yes"
    assert summary["question_type"] == "yes_no"
    assert summary["generated_code"] == "def program(input_scene): pass"
    assert summary["flowchart_path"] == "flowcharts/omni3d_0_flowchart.png"
    assert summary["float_relative_error"] == 0.01
    assert summary["float_mra"] == 1.0
    assert summary["numeric_relative_error"] == 0.01
    assert summary["numeric_mra"] == 1.0
    assert summary["numeric_score"] == 1.0
    assert "object_3d_boxes" not in summary
    assert "visual_clue" not in summary
    assert "generated_response" not in summary

