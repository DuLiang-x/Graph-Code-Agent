import argparse
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

DEMO_SCRIPT = TEST_ROOT / "scripts" / "demo_extract_3d_positions.py"
DEMO_SPEC = importlib.util.spec_from_file_location("demo_extract_3d_positions", DEMO_SCRIPT)
demo_extract_3d_positions = importlib.util.module_from_spec(DEMO_SPEC)
DEMO_SPEC.loader.exec_module(demo_extract_3d_positions)

from object_3d_extraction import Object3DExtractionConfig, Object3DLocator
from object_3d_extraction.depth_module import unproject_to_3D
from object_3d_extraction.object_3d_locator import detection_prompts_for_object, filter_count_instance_candidates, maybe_fallback_mask, parse_candidate_index, parse_object_relation_context, rank_detection_candidates, validate_vlm_selection
from object_3d_extraction.utils import extract_object_names_from_question_options, save_debug_visuals


class FakeDetectionModule:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, category):
        return self.detections.get(category, [])

    def run_segmentation(self, image, box2d):
        mask = np.zeros((image.height, image.width), dtype=np.float32)
        x1, y1, x2, y2 = box2d
        mask[y1:y2, x1:x2] = 1.0
        return mask




class FakeOrientationModule:
    def run_orientation_estimation(self, image, bbox=None):
        return {
            "orientation": [0.0, 0.0, -1.0],
            "orientation_angles": [0.0, 0.0, 0.0],
        }


class FakeDepthModule:
    def run_depth_estimation(self, image):
        yy, xx = np.indices((image.height, image.width))
        return (2.0 + 0.01 * xx + 0.02 * yy).astype(np.float32)

    def unproject_to_3D(self, image, depth, segment_mask):
        return unproject_to_3D(image, depth, segment_mask, min_depth_points=1)


def make_locator(detections, vlm_model=None, use_vlm_refinement=False, mask_fallback_mode=None):
    cfg = Object3DExtractionConfig(min_mask_area=1, min_depth_points=1)
    cfg.detection.use_vlm_refinement = use_vlm_refinement
    if mask_fallback_mode is not None:
        cfg.mask_fallback_mode = mask_fallback_mode
    return Object3DLocator(
        config=cfg,
        device="cpu",
        detection_module=FakeDetectionModule(detections),
        depth_module=FakeDepthModule(),
        orientation_module=FakeOrientationModule(),
        vlm_model=vlm_model,
    )


def test_missing_image_path_has_clear_error():
    locator = make_locator({})

    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        locator.extract("missing_image.jpg", ["person"])


def test_empty_object_names_has_clear_error():
    locator = make_locator({})
    image = Image.new("RGB", (16, 16), color="white")

    with pytest.raises(ValueError, match="object_names must contain"):
        locator.extract(image, [])


def test_detection_failure_returns_error_for_object():
    locator = make_locator({"person": []})
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(image, ["person"])

    assert "error" in result["person"]
    assert "No detection" in result["person"]["error"]


def test_success_returns_position_orientation_and_3d_box():
    locator = make_locator(
        {
            "chair": [
                {
                    "box2d": [4, 4, 12, 12],
                    "score": 0.9,
                }
            ]
        }
    )
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(image, ["chair"])

    item = result["chair"]
    assert isinstance(item["position"], list)
    assert len(item["position"]) == 3
    assert all(isinstance(value, float) for value in item["position"])
    assert item["box2d"] == [4, 4, 12, 12]
    assert item["bbox_wh"] == [8, 8]
    assert item["mask_area"] == 64
    assert item["num_points"] > 0
    assert item["orientation"] == [0.0, 0.0, -1.0]
    assert item["orientation_angles"] == [0.0, 0.0, 0.0]
    assert item["prompt_used"] == "chair"
    assert "orientation_abr" not in item
    assert isinstance(item["box3d_size"], list)
    assert len(item["box3d_size"]) == 3
    assert all(isinstance(value, float) for value in item["box3d_size"])
    assert all(value >= 0.0 for value in item["box3d_size"])
    assert len(item["box3d_center"]) == 3
    assert len(item["box3d_min"]) == 3
    assert len(item["box3d_max"]) == 3


def test_count_question_expands_count_target_into_indexed_instances():
    locator = make_locator(
        {
            "handles": [
                {"box2d": [2, 2, 6, 6], "score": 0.90},
                {"box2d": [10, 2, 14, 6], "score": 0.88},
                {"box2d": [18, 2, 22, 6], "score": 0.86},
            ],
            "cabinets": [
                {"box2d": [0, 0, 30, 15], "score": 0.80},
            ],
        }
    )
    image = Image.new("RGB", (32, 20), color="white")

    result = locator.extract(
        image,
        ["handles", "cabinets"],
        question="How many handles are on the cabinets?",
        question_type="numeric_ct",
    )

    assert result["handles"]["counting_target"] is True
    assert result["handles"]["counting_instances"] == ["handle_1", "handle_2", "handle_3"]
    assert result["handles"]["error"] == "Count target expanded into indexed instances"
    assert "position" not in result["handles"]
    assert all(name in result for name in ["handle_1", "handle_2", "handle_3"])
    assert result["handle_1"]["counting_source_object"] == "handles"
    assert result["handle_1"]["selection_decision"] == "count_instance"
    assert "cabinets" in result and "position" in result["cabinets"]


def test_non_count_question_keeps_single_target_node():
    locator = make_locator(
        {
            "handles": [
                {"box2d": [2, 2, 6, 6], "score": 0.90},
                {"box2d": [10, 2, 14, 6], "score": 0.88},
            ],
        }
    )
    image = Image.new("RGB", (32, 20), color="white")

    result = locator.extract(image, ["handles"], question="Where are the handles?", question_type="generic")

    assert "handles" in result
    assert "handle_1" not in result
    assert "position" in result["handles"]


def test_count_question_uses_vlm_multi_select_when_available():
    vlm = FakeObjectExtractionVLM(["[0, 1, 2]"])
    locator = make_locator(
        {
            "handles": [
                {"box2d": [2, 2, 6, 6], "score": 0.90},
                {"box2d": [10, 2, 14, 6], "score": 0.88},
                {"box2d": [18, 2, 22, 6], "score": 0.86},
            ],
        },
        vlm_model=vlm,
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (32, 20), color="white")

    result = locator.extract(
        image,
        ["handles"],
        question="How many handles are on the cabinets?",
        question_type="numeric_ct",
    )

    assert result["handles"]["counting_selection_source"] == "vlm_multi_select"
    assert result["handles"]["counting_vlm_selected_indices"] == [0, 1, 2]
    assert result["handles"]["counting_instances"] == ["handle_1", "handle_2", "handle_3"]
    assert result["handle_1"]["counting_selection_source"] == "vlm_multi_select"
    assert result["handle_1"]["counting_vlm_response"] == "[0, 1, 2]"
    prompt = vlm.calls[0][0][0]["content"][2]["text"]
    assert "Return only a Python-style list of integer indices" in prompt
    assert "[0, 2, 5]" in prompt
    assert "Lightbulbs: select individual bulbs" in prompt
    assert "Post-it notes: select individual notes" in prompt
    assert "Do not select the same instance twice" in prompt
    assert "Do not select a large box that contains multiple counted instances" in prompt


def test_count_question_invalid_vlm_response_falls_back_to_rule_instances():
    vlm = FakeObjectExtractionVLM(["INVALID"])
    locator = make_locator(
        {
            "handles": [
                {"box2d": [2, 2, 6, 6], "score": 0.90},
                {"box2d": [10, 2, 14, 6], "score": 0.88},
            ],
        },
        vlm_model=vlm,
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (32, 20), color="white")

    result = locator.extract(
        image,
        ["handles"],
        question="How many handles are on the cabinets?",
        question_type="numeric_ct",
    )

    assert result["handles"]["counting_selection_source"] == "rule_fallback"
    assert result["handles"]["counting_vlm_response"] == "INVALID"
    assert result["handles"]["counting_instances"] == ["handle_1", "handle_2"]


def test_count_filter_removes_group_box_containing_smaller_instances():
    candidates = [
        {"box2d": [0, 0, 80, 80], "score": 0.95, "prompt": "post-it notes", "candidate_index": 0},
        {"box2d": [10, 10, 20, 20], "score": 0.70, "prompt": "post-it notes", "candidate_index": 1},
        {"box2d": [30, 10, 40, 20], "score": 0.68, "prompt": "post-it notes", "candidate_index": 2},
    ]

    kept, rejected = filter_count_instance_candidates(candidates, candidates, "post-it notes", max_instances=10)

    assert [item["candidate_index"] for item in kept] == [1, 2]
    assert rejected[0]["counting_filter_reason"] == "contained_group_box"


def test_count_filter_removes_duplicate_iou_and_container_object():
    candidates = [
        {"box2d": [0, 0, 20, 20], "score": 0.95, "prompt": "lightbulbs", "candidate_index": 0},
        {"box2d": [1, 1, 21, 21], "score": 0.94, "prompt": "lightbulbs", "candidate_index": 1},
        {"box2d": [0, 0, 80, 80], "score": 0.90, "prompt": "chandelier", "candidate_index": 2},
    ]

    kept, rejected = filter_count_instance_candidates(candidates, candidates, "lightbulbs", max_instances=10)

    assert [item["candidate_index"] for item in kept] == [0]
    assert {item["counting_filter_reason"] for item in rejected} == {"duplicate_iou", "container_object"}


def test_success_records_sam_mask_usage_fields():
    locator = make_locator(
        {"chair": [{"box2d": [4, 4, 12, 12], "score": 0.9}]}
    )
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(image, ["chair"])

    item = result["chair"]
    assert item["mask_used_for_3d"] == "sam"
    assert item["sam_mask_area"] == item["final_mask_area"] == 64
    assert item["mask_fallback_reason"] is None


def test_default_auto_fallback_records_distinct_sam_and_final_mask_areas():
    locator = make_locator(
        {"white coffee table": [{"box2d": [0, 20, 100, 100], "score": 0.9}]}
    )
    image = Image.new("RGB", (100, 100), color="white")

    result = locator.extract(image, ["white coffee table"])

    item = result["white coffee table"]
    assert item["mask_used_for_3d"] == "fallback"
    assert item["mask_fallback_reason"] == "entity_box_too_large"
    assert item["sam_mask_area"] > item["final_mask_area"]


def test_mask_fallback_off_uses_sam_mask_for_large_entity_box():
    locator = make_locator(
        {"white coffee table": [{"box2d": [0, 20, 100, 100], "score": 0.9}]},
        mask_fallback_mode="off",
    )
    image = Image.new("RGB", (100, 100), color="white")

    result = locator.extract(image, ["white coffee table"])

    item = result["white coffee table"]
    assert item["mask_used_for_3d"] == "sam"
    assert item["mask_fallback_reason"] is None
    assert item["sam_mask_area"] == item["final_mask_area"]


def test_save_debug_visuals_writes_sam_and_final_overlays(tmpdir):
    image = Image.new("RGB", (16, 16), color="white")
    sam_mask = np.zeros((16, 16), dtype=np.float32)
    sam_mask[2:14, 2:14] = 1.0
    final_mask = np.zeros((16, 16), dtype=np.float32)
    final_mask[5:10, 5:10] = 1.0

    save_debug_visuals(
        image,
        "white coffee table",
        [2, 2, 14, 14],
        final_mask,
        str(tmpdir),
        sam_mask=sam_mask,
        mask_used_for_3d="fallback",
    )

    assert tmpdir.join("white_coffee_table_sam_mask.png").check()
    assert tmpdir.join("white_coffee_table_sam_overlay.png").check()
    assert tmpdir.join("white_coffee_table_mask.png").check()
    assert tmpdir.join("white_coffee_table_overlay.png").check()


def test_extract_visualize_writes_3d_debug_panel(tmpdir):
    locator = make_locator(
        {"chair": [{"box2d": [4, 4, 12, 12], "score": 0.9}]},
        mask_fallback_mode="off",
    )
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(
        image,
        ["chair"],
        visualize=True,
        save_dir=str(tmpdir),
        question="Where is the chair?",
    )

    assert "chair" in result
    assert tmpdir.join("3d_aabb.png").check()
    assert tmpdir.join("3d_debug_panel.png").check()
    assert locator.last_3d_visualization_paths["aabb_vis_path"].endswith("3d_aabb.png")
    assert locator.last_3d_visualization_paths["panel_path"].endswith("3d_debug_panel.png")


def test_unproject_returns_3d_box_fields():
    image = Image.new("RGB", (8, 8), color="white")
    depth = np.ones((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[3, 3] = 1.0

    result = unproject_to_3D(image, depth, mask, min_depth_points=1)

    assert result["position"]
    assert result["box3d_size"] == [0.0, 0.0, 0.0]
    assert result["box3d_center"] == result["position"]
    assert "orientation_abr" not in result
    assert "orientation_error" not in result


def test_extract_object_names_from_question_options():
    question = (
        "Based on these four images (image 1, 2, 3, and 4) showing the black sneaker "
        "from different viewpoints (front, left, back, and right), with each camera "
        "aligned with room walls and partially capturing the surroundings: From the "
        "viewpoint presented in image 2, what is to the right of the black sneaker? "
        "A. TV B. Wooden dining table C. Light purple sofa D. Brown curtains and windows"
    )

    objects = extract_object_names_from_question_options(question)

    assert objects == [
        "black sneaker",
        "tv",
        "wooden dining table",
        "light purple sofa",
        "brown curtains and windows",
    ]


def test_omni3d_sample_uses_image_filename_and_question_index():
    args = type("Args", (), {"image": None, "image_root": "/data/datasets/Omni3D-Bench/images"})()
    sample = {
        "question_index": 12,
        "image_filename": "ARKitScenes/Training/43828442/91339.246_00000463.jpg",
    }

    assert demo_extract_3d_positions.get_sample_key(sample, 0) == "omni3d_12"
    assert demo_extract_3d_positions.resolve_image_path(args, sample) == (
        "/data/datasets/Omni3D-Bench/images/"
        "ARKitScenes/Training/43828442/91339.246_00000463.jpg"
    )


def test_omni3d_question_object_name_extraction():
    samples = [
        (
            "What is the ratio of the height of the fireplace to the combined height "
            "of the coffee table and the sofa to the right of the coffee table?",
            ["fireplace", "coffee table", "sofa"],
        ),
        (
            "If the coffee table is 2m long, how long is the sofa to the right of it in meters?",
            ["coffee table", "sofa"],
        ),
        (
            "Which object is closer to the fireplace: the sofa or the white coffee table? "
            "Options: {sofa, coffee table}",
            ["sofa", "coffee table", "fireplace", "white coffee table"],
        ),
    ]

    for question, expected in samples:
        sample = {"question": question}
        assert demo_extract_3d_positions.resolve_object_names(sample, use_vlm_object_extraction=False)["objects"] == expected



class FakeObjectExtractionVLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def process_messages(self, messages, max_new_tokens=128):
        self.calls.append((messages, max_new_tokens))
        return self.responses.pop(0)


def test_classify_question_type_for_omni3d_categories():
    assert demo_extract_3d_positions.classify_question_type({
        "question": "How many handles are on the cabinets?",
        "answer_type": "int",
        "answer": 11,
    }) == "numeric_ct"
    assert demo_extract_3d_positions.classify_question_type({
        "question": "How many stools are needed to match the chair height?",
        "answer_type": "float",
        "answer": 1.8,
    }) == "numeric_other"
    assert demo_extract_3d_positions.classify_question_type({
        "question": "How many objects with the volume of the combined volume of the two bedside tables would fit in the bed?",
        "answer_type": "float",
        "answer": 9.658,
    }) == "numeric_other"
    assert demo_extract_3d_positions.classify_question_type({
        "question": "What is the ratio of the fireplace height to the sofa height?",
        "answer_type": "float",
        "answer": 0.94,
    }) == "numeric_other"
    assert demo_extract_3d_positions.classify_question_type({
        "question": "Are the number of cabinets greater than the number of windows visible?",
        "answer_type": "str",
        "answer": "yes",
    }) == "yes_no"
    assert demo_extract_3d_positions.classify_question_type({
        "question": "Which object is closer to the fireplace: the sofa or the coffee table?",
        "answer_type": "str",
        "answer": "sofa",
    }) == "choice_object"


def test_vlm_object_extraction_prompt_uses_question_type_rules(tmpdir):
    image_path = Path(str(tmpdir)) / "scene.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    sample = {
        "question": "How many handles are on the cabinets?",
        "answer_type": "int",
        "answer": 11,
    }
    vlm = FakeObjectExtractionVLM(["[handles, cabinets]"])

    info = demo_extract_3d_positions.resolve_object_names(sample, image=str(image_path), vlm_model=vlm)

    prompt = vlm.calls[0][0][0]["content"][1]["text"]
    assert info["question_type"] == "numeric_ct"
    assert info["objects"] == ["handles", "cabinets"]
    assert "Question type: numeric count" in prompt
    assert "all visible instances" in prompt
    assert "[Detect] [handles, cabinets]" in prompt


def test_vlm_object_extraction_preferred_over_rule(tmpdir):
    image_path = Path(str(tmpdir)) / "scene.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    sample = {"question": "What is the ratio of the height of the fireplace to the sofa?"}
    vlm = FakeObjectExtractionVLM(["[fireplace, coffee table, sofa]"])

    info = demo_extract_3d_positions.resolve_object_names(sample, image=str(image_path), vlm_model=vlm)

    assert info["objects"] == ["fireplace", "coffee table", "sofa"]
    assert info["method"] == "vlm"
    assert info["vlm_extracted_objects"] == ["fireplace", "coffee table", "sofa"]
    assert info["object_extraction_response"] == "[fireplace, coffee table, sofa]"


def test_vlm_object_extraction_aux_retry(tmpdir):
    image_path = Path(str(tmpdir)) / "scene.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    sample = {"question": "Which object is closer to the fireplace: the sofa or the coffee table?"}
    vlm = FakeObjectExtractionVLM(["fireplace, sofa", "[fireplace, sofa, coffee table]"])

    info = demo_extract_3d_positions.resolve_object_names(sample, image=str(image_path), vlm_model=vlm)

    assert info["objects"] == ["fireplace", "sofa", "coffee table"]
    assert info["method"] == "vlm"
    assert len(vlm.calls) == 2
    assert "Previous response: fireplace, sofa" in vlm.calls[1][0][0]["content"][1]["text"]


def test_vlm_object_extraction_falls_back_to_rule(tmpdir):
    image_path = Path(str(tmpdir)) / "scene.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    sample = {"question": "What is the ratio of the height of the fireplace to the sofa?"}
    vlm = FakeObjectExtractionVLM(["no list", "still no list"])

    info = demo_extract_3d_positions.resolve_object_names(sample, image=str(image_path), vlm_model=vlm)

    assert info["objects"] == ["fireplace", "sofa"]
    assert info["method"] == "rule_fallback"
    assert info["vlm_extracted_objects"] == []
    assert info["rule_extracted_objects"] == ["fireplace", "sofa"]


def test_no_vlm_object_extraction_does_not_call_vlm(tmpdir):
    image_path = Path(str(tmpdir)) / "scene.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    sample = {"question": "What is the ratio of the height of the fireplace to the sofa?"}
    vlm = FakeObjectExtractionVLM(["[wrong]"])

    info = demo_extract_3d_positions.resolve_object_names(
        sample,
        image=str(image_path),
        vlm_model=vlm,
        use_vlm_object_extraction=False,
    )

    assert info["objects"] == ["fireplace", "sofa"]
    assert info["method"] == "rule"
    assert vlm.calls == []

def test_detection_prompt_fallback_uses_simplified_object_name():
    locator = make_locator(
        {
            "rightmost sofa": [],
            "sofa": [
                {
                    "box2d": [4, 4, 12, 12],
                    "score": 0.8,
                }
            ],
        }
    )
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(image, ["rightmost sofa"])

    item = result["rightmost sofa"]
    assert "error" not in item
    assert item["prompt_used"] == "sofa"
    assert item["score"] == 0.8


def test_detection_failure_records_prompts_tried():
    locator = make_locator({"white coffee table": [], "coffee table": [], "white table": [], "table": []})
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(image, ["white coffee table"])

    item = result["white coffee table"]
    assert item["error"] == "No detection for object: white coffee table"
    assert item["prompts_tried"] == ["white coffee table", "coffee table", "white table", "table"]


def test_detection_prompt_simplification():
    assert detection_prompts_for_object("rightmost sofa") == ["rightmost sofa", "sofa"]
    assert detection_prompts_for_object("white coffee table") == ["white coffee table", "coffee table", "white table", "table"]
    assert detection_prompts_for_object("glass table") == ["glass table", "transparent table", "glass coffee table", "clear table"]
    assert detection_prompts_for_object("tv") == ["tv", "television"]
    assert detection_prompts_for_object("gray chair") == ["gray chair", "chair"]
    assert detection_prompts_for_object("black chair") == ["black chair", "chair"]
    assert detection_prompts_for_object("translucent cube") == [
        "translucent cube",
        "transparent cube",
        "clear cube",
        "see-through cube",
        "cube",
    ]


def test_relation_context_detection_prompts_keep_target_not_reference():
    prompts = detection_prompts_for_object("circular table under the tv")

    assert prompts == ["circular table", "round table", "table"]
    assert "circular table under the tv" not in prompts
    assert "tv" not in prompts
    assert "television" not in prompts
    white_prompts = detection_prompts_for_object("white table next to sofa")
    assert "white table" in white_prompts
    assert "sofa" not in white_prompts
    glass_prompts = detection_prompts_for_object("glass table in front of sofa")
    assert "glass table" in glass_prompts
    assert "transparent table" in glass_prompts
    assert "clear table" in glass_prompts


def test_parse_object_relation_context():
    assert parse_object_relation_context("circular table under the tv") == {
        "target_phrase": "circular table",
        "relation": "under",
        "relation_context": "under the tv",
        "reference_object": "tv",
    }
    assert parse_object_relation_context("black table to the right of the chair") == {
        "target_phrase": "black table",
        "relation": "right of",
        "relation_context": "to the right of the chair",
        "reference_object": "chair",
    }
    assert parse_object_relation_context("white table next to the sofa") == {
        "target_phrase": "white table",
        "relation": "next to",
        "relation_context": "next to the sofa",
        "reference_object": "sofa",
    }

def test_rule_ranker_avoids_large_area_box_for_non_area_object():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [0, 40, 100, 100], "score": 0.9, "prompt": "white coffee table"},
        {"box2d": [10, 50, 45, 75], "score": 0.55, "prompt": "table"},
    ]

    selected = rank_detection_candidates(image, "white coffee table", candidates, {})

    assert selected["box2d"] == [10, 50, 45, 75]
    assert selected["prompt"] == "table"


def test_rule_ranker_selects_rightmost_candidate_when_quality_is_reasonable():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [5, 10, 25, 40], "score": 0.9, "prompt": "sofa"},
        {"box2d": [70, 10, 95, 40], "score": 0.75, "prompt": "sofa"},
    ]

    selected = rank_detection_candidates(image, "rightmost sofa", candidates, {})

    assert selected["box2d"] == [70, 10, 95, 40]


def test_leftmost_relation_keeps_extreme_large_chair_candidate():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [70, 30, 99, 99], "score": 0.62, "prompt": "left-most chair"},
        {"box2d": [0, 0, 26, 98], "score": 0.62, "prompt": "chair"},
    ]

    selected = rank_detection_candidates(image, "left-most chair", candidates, {})

    assert selected["box2d"] == [0, 0, 26, 98]
    assert "broad_entity_penalty" in selected["rank_reasons"]


def test_shape_prompt_adds_bonus_for_circular_table_prompts():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [10, 10, 50, 50], "score": 0.50, "prompt": "round table"},
        {"box2d": [55, 10, 95, 50], "score": 0.50, "prompt": "table"},
    ]

    selected = rank_detection_candidates(image, "circular table", candidates, {})

    assert selected["box2d"] == [10, 10, 50, 50]
    assert "shape_prompt" in selected["rank_reasons"]


def test_material_prompt_adds_bonus_for_glass_table_prompts():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [10, 10, 50, 50], "score": 0.50, "prompt": "transparent table"},
        {"box2d": [55, 10, 95, 50], "score": 0.50, "prompt": "table"},
    ]

    selected = rank_detection_candidates(image, "glass table", candidates, {})

    assert selected["box2d"] == [10, 10, 50, 50]
    assert "material_prompt" in selected["rank_reasons"]


def test_color_prompt_adds_bonus_for_same_category_attributes():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [10, 10, 45, 45], "score": 0.50, "prompt": "gray chair"},
        {"box2d": [55, 10, 90, 45], "score": 0.50, "prompt": "black chair"},
    ]

    gray = rank_detection_candidates(image, "gray chair", candidates, {})
    black = rank_detection_candidates(image, "black chair", candidates, {})

    assert gray["box2d"] == [10, 10, 45, 45]
    assert "color_prompt" in gray["rank_reasons"]
    assert black["box2d"] == [55, 10, 90, 45]
    assert "color_prompt" in black["rank_reasons"]


def test_transparency_prompt_adds_bonus_for_translucent_cube_prompts():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [10, 10, 45, 45], "score": 0.50, "prompt": "transparent cube"},
        {"box2d": [55, 10, 90, 45], "score": 0.50, "prompt": "cube"},
    ]

    selected = rank_detection_candidates(image, "translucent cube", candidates, {})

    assert selected["box2d"] == [10, 10, 45, 45]
    assert "material_prompt" in selected["rank_reasons"]


def test_entity_ranker_prefers_tight_white_coffee_table_over_large_carpet_box():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [0, 35, 100, 100], "score": 0.9, "prompt": "white coffee table"},
        {"box2d": [10, 45, 45, 72], "score": 0.58, "prompt": "white table"},
    ]

    selected = rank_detection_candidates(image, "white coffee table", candidates, {})

    assert selected["box2d"] == [10, 45, 45, 72]
    assert "large_entity_penalty" not in selected["rank_reasons"]


def test_large_entity_ranker_allows_exact_bed_box_despite_large_area():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [0, 0, 95, 95], "score": 0.92, "prompt": "bed"},
        {"box2d": [0, 35, 25, 70], "score": 0.18, "prompt": "bed"},
    ]

    selected = rank_detection_candidates(image, "bed", candidates, {})

    assert selected["box2d"] == [0, 0, 95, 95]
    assert "large_entity_tolerated" in selected["rank_reasons"]
    assert "broad_entity_tolerated" in selected["rank_reasons"]


def test_area_ranker_prefers_low_wide_carpet_over_high_platform():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [45, 20, 75, 42], "score": 0.65, "prompt": "carpet"},
        {"box2d": [0, 58, 95, 98], "score": 0.4, "prompt": "carpet"},
    ]

    selected = rank_detection_candidates(image, "carpet", candidates, {})

    assert selected["box2d"] == [0, 58, 95, 98]
    assert "wide_area_object" in selected["rank_reasons"]
    assert "low_area_object" in selected["rank_reasons"]


def test_black_table_ranker_penalizes_large_fallback_table_box():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [0, 20, 100, 100], "score": 0.9, "prompt": "table"},
        {"box2d": [25, 65, 80, 98], "score": 0.62, "prompt": "black table"},
    ]

    selected = rank_detection_candidates(image, "black table", candidates, {})

    assert selected["box2d"] == [25, 65, 80, 98]
    assert "color_prompt" in selected["rank_reasons"]


class InvalidVLM:
    def process_messages(self, messages, max_new_tokens=32):
        assert messages[0]["content"][0]["type"] == "image"
        assert messages[0]["content"][1]["type"] == "image"
        return "INVALID"


def test_vlm_invalid_response_falls_back_to_rule_ranker(tmpdir):
    locator = make_locator(
        {
            "white coffee table": [
                {"box2d": [0, 35, 100, 100], "score": 0.9, "prompt": "white coffee table"},
                {"box2d": [10, 45, 45, 72], "score": 0.58, "prompt": "white coffee table"},
            ]
        },
        vlm_model=InvalidVLM(),
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (100, 100), color="white")

    result = locator.extract(image, ["white coffee table"], visualize=True, save_dir=str(tmpdir))

    item = result["white coffee table"]
    assert item["box2d"] == [10, 45, 45, 72]
    assert item["candidate_rank_reason"] == "rule_ranker"
    assert item["candidate_overlay_path"].endswith("detection_candidates_white_coffee_table_overlay.png")


def test_entity_large_box_triggers_mask_fallback():
    image = Image.new("RGB", (100, 100), color="white")
    mask = np.ones((100, 100), dtype=np.float32)

    fallback_mask, reason = maybe_fallback_mask(image, "white coffee table", [0, 20, 100, 100], mask)

    assert reason == "entity_box_too_large"
    assert int((fallback_mask > 0.5).sum()) < int(mask.sum())


class FakeVLM:
    def __init__(self, response="1"):
        self.response = response
        self.messages = None
        self.max_new_tokens = None

    def process_messages(self, messages, max_new_tokens=32):
        self.messages = messages
        self.max_new_tokens = max_new_tokens
        assert messages[0]["content"][0]["type"] == "image"
        return self.response


def test_vlm_refinement_selects_mocked_candidate_index():
    vlm = FakeVLM()
    locator = make_locator(
        {
            "chair": [
                {"box2d": [1, 1, 5, 5], "score": 0.9},
                {"box2d": [8, 8, 12, 12], "score": 0.75},
            ]
        },
        vlm_model=vlm,
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(image, ["chair"])

    item = result["chair"]
    assert item["box2d"] == [8, 8, 12, 12]
    assert item["candidate_rank_reason"] == "vlm_refinement"
    assert item["rule_selected_index"] == 0
    assert item["vlm_selected_index"] == 1
    assert item["final_selected_index"] == 1
    assert item["selection_decision"] == "vlm_refinement"
    assert vlm.max_new_tokens == 32
    content = vlm.messages[0]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "image"
    assert content[2]["type"] == "text"
    prompt = content[2]["text"]
    assert "Candidate metadata table:" in prompt
    assert "index | prompt | box2d | center | area_ratio | dino_score | rank_score | rank_reasons" in prompt
    assert "Coordinate and score notes:" in prompt
    assert "Larger x means further right in the image; larger y means lower in the image" in prompt
    assert "dino_score, rank_score, and rank_reasons are supporting evidence only" in prompt
    assert "approx_depth" not in prompt
    assert "Candidate 0 is the rule-ranked best candidate, but you may choose another candidate if it better matches the exact target object category and question context." in prompt
    assert "not just the most visually obvious object of the category" in prompt
    assert "Target object phrase:" in prompt
    assert "Main target:" in prompt
    assert "Relation context:" in prompt
    assert "Reference object:" in prompt
    assert "color, material, shape, relative position, nearby objects, and role in the scene" in prompt
    assert '"white coffee table", "circular table", "black table", "person wearing a hat", "left chair", or "closer sofa"' in prompt
    assert "under the tv" in prompt
    assert "it does not mean choose the TV" in prompt
    assert "Use the full-image overlay first" in prompt
    assert "Use the crop grid only as supporting evidence" in prompt
    assert "Object-specific rules:" in prompt
    assert "avoid boxes that include large carpet/floor/background regions or multiple objects" in prompt
    assert "avoid partial boxes that only cover the top, seat, backrest, leg, or one component" in prompt
    assert "prefer the large low horizontal floor-covering region" in prompt
    assert "Similar-object rules:" in prompt
    assert "chair, stool, bench, sofa, couch, ottoman, and seat" in prompt
    assert 'Do not select a stool for "chair"' in prompt
    assert 'Do not select a chair for "stool"' in prompt
    assert 'Do not select a chair for "bench", "sofa", "couch", or "ottoman"' in prompt
    assert 'Do not select an ottoman for "stool" or a stool for "ottoman"' in prompt
    assert "chair: usually has a backrest" in prompt
    assert "stool: usually has no backrest" in prompt
    assert "bench: usually elongated" in prompt
    assert "sofa/couch: usually larger" in prompt
    assert "ottoman: usually a low padded seat" in prompt
    assert "Treat words such as rightmost, leftmost, topmost, and bottommost as part of the target object phrase" in prompt
    assert 'For "rightmost chair" or similar targets, choose the rightmost complete candidate' in prompt
    assert "Do not choose a larger, clearer, or more central candidate" in prompt
    assert "Counting and instance rules:" in prompt
    assert "Do not select the same instance twice" in prompt
    assert "Do not select a large box that contains multiple counted instances" in prompt
    assert "Badcase-guided examples:" in prompt
    assert 'for "circular table under the TV"' in prompt
    assert 'for "translucent cube"' in prompt
    assert 'for "gray chair" versus "black chair"' in prompt
    assert 'for "brown chair" versus "black chair"' in prompt
    assert "camera is viewpoint" in prompt
    assert 'For "glass table", choose the actual glass/transparent table' in prompt
    assert "not a black plastic table or ordinary dark table" in prompt
    assert "Treat shape words such as circular, round, square, rectangular, and oval" in prompt
    assert 'For "circular table" or "round table", choose the round/circular table' in prompt
    assert "Return only one integer index" in prompt
    assert "return INVALID" in prompt


def test_rightmost_relation_prefers_quality_candidate_over_extreme_partial_box():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [80, 10, 100, 40], "score": 0.20, "prompt": "rightmost stool"},
        {"box2d": [50, 60, 95, 95], "score": 0.70, "prompt": "rightmost stool"},
        {"box2d": [45, 62, 93, 97], "score": 0.68, "prompt": "stool"},
    ]

    selected = rank_detection_candidates(image, "rightmost stool", candidates)

    assert selected["box2d"] == [50, 60, 95, 95]


def test_vlm_refinement_rejects_relation_mismatch_for_rightmost_object():
    locator = make_locator(
        {
            "rightmost chair": [
                {"box2d": [70, 10, 90, 50], "score": 0.70, "prompt": "rightmost chair"},
                {"box2d": [10, 10, 40, 50], "score": 0.65, "prompt": "rightmost chair"},
            ]
        },
        vlm_model=FakeVLM(response="1"),
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (100, 100), color="white")

    result = locator.extract(image, ["rightmost chair"])

    item = result["rightmost chair"]
    assert item["box2d"] == [70, 10, 90, 50]
    assert item["candidate_rank_reason"] == "rule_ranker"
    assert item["rule_selected_index"] == 0
    assert item["vlm_selected_index"] == 1
    assert item["final_selected_index"] == 0
    assert item["selection_decision"] == "rule_ranker_vlm_rejected"
    assert item["selection_reject_reason"] == "relation_mismatch"


def test_vlm_selection_accepts_relation_target_despite_smaller_box():
    candidates = [
        {"box2d": [0, 0, 100, 70], "score": 0.87, "rank_score": 0.95, "prompt": "tv"},
        {"box2d": [45, 62, 70, 75], "score": 0.58, "rank_score": 0.76, "prompt": "circular table"},
    ]

    allowed, reject_reason = validate_vlm_selection("circular table under the tv", candidates[1], candidates)

    assert allowed is True
    assert reject_reason is None


def test_vlm_selection_rejects_reference_object_for_relation_target():
    candidates = [
        {"box2d": [0, 0, 100, 70], "score": 0.87, "rank_score": 0.95, "prompt": "tv"},
        {"box2d": [45, 62, 70, 75], "score": 0.58, "rank_score": 0.76, "prompt": "circular table"},
    ]

    allowed, reject_reason = validate_vlm_selection("circular table under the tv", candidates[0], candidates)

    assert allowed is False
    assert reject_reason == "reference_object_selected"


def test_vlm_selection_rejects_weak_phrase_low_score_candidate():
    candidates = [
        {"box2d": [20, 70, 45, 90], "score": 0.80, "rank_score": 0.98, "prompt": "small stool"},
        {"box2d": [2, 58, 15, 74], "score": 0.05, "rank_score": 0.15, "prompt": "small"},
    ]

    allowed, reject_reason = validate_vlm_selection("small stool", candidates[1], candidates)

    assert allowed is False
    assert reject_reason == "weak_phrase_low_score"


def test_vlm_selection_accepts_larger_translucent_cube_attribute_candidate():
    candidates = [
        {"box2d": [0, 0, 10, 10], "score": 0.70, "rank_score": 0.88, "prompt": "translucent cube"},
        {"box2d": [20, 20, 40, 40], "score": 0.50, "rank_score": 0.68, "prompt": "transparent cube"},
    ]

    allowed, reject_reason = validate_vlm_selection("translucent cube", candidates[1], candidates)

    assert allowed is True
    assert reject_reason is None


def test_vlm_selection_still_rejects_obviously_huge_translucent_cube_candidate():
    candidates = [
        {"box2d": [0, 0, 10, 10], "score": 0.70, "rank_score": 0.88, "prompt": "translucent cube"},
        {"box2d": [0, 0, 90, 90], "score": 0.50, "rank_score": 0.68, "prompt": "transparent cube"},
    ]

    allowed, reject_reason = validate_vlm_selection("translucent cube", candidates[1], candidates)

    assert allowed is False
    assert reject_reason == "entity_box_too_large"


def test_vlm_refinement_allows_material_specific_lower_rank_candidate():
    locator = make_locator(
        {
            "glass table": [
                {"box2d": [55, 55, 95, 95], "score": 0.70, "prompt": "glass table"},
                {"box2d": [10, 10, 45, 45], "score": 0.45, "prompt": "transparent table"},
            ]
        },
        vlm_model=FakeVLM(response="1"),
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (100, 100), color="white")

    result = locator.extract(image, ["glass table"])

    item = result["glass table"]
    assert item["box2d"] == [10, 10, 45, 45]
    assert item["candidate_rank_reason"] == "vlm_refinement"
    assert item["selection_decision"] == "vlm_refinement"
    assert item["selection_reject_reason"] is None


def test_vlm_validation_accepts_lower_rank_attribute_candidate():
    candidates = [
        {"box2d": [27, 72, 71, 98], "score": 0.47, "rank_score": 0.85, "prompt": "table"},
        {"box2d": [38, 35, 58, 55], "score": 0.20, "rank_score": 0.30, "prompt": "black table"},
    ]

    allowed, reject_reason = validate_vlm_selection("black table", candidates[1], candidates)

    assert allowed is True
    assert reject_reason is None


def test_vlm_refinement_rejects_partial_cabinet_box():
    locator = make_locator(
        {
            "cabinet": [
                {"box2d": [40, 0, 75, 60], "score": 0.54, "prompt": "cabinet"},
                {"box2d": [41, 0, 75, 31], "score": 0.35, "prompt": "cabinet"},
            ]
        },
        vlm_model=FakeVLM(),
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (100, 100), color="white")

    result = locator.extract(image, ["cabinet"])

    item = result["cabinet"]
    assert item["box2d"] == [40, 0, 75, 60]
    assert item["candidate_rank_reason"] == "rule_ranker"
    assert item["rule_selected_index"] == 0
    assert item["vlm_selected_index"] == 1
    assert item["final_selected_index"] == 0
    assert item["selection_decision"] == "rule_ranker_vlm_rejected"
    assert item["selection_reject_reason"] in {"rank_score_too_low", "partial_entity_box"}

def test_vlm_refinement_invalid_response_falls_back_to_first_candidate():
    assert parse_candidate_index("not an index", 3) == 0
    assert parse_candidate_index("9", 3) == 0


def test_cli_default_vlm_model_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo_extract_3d_positions.py"])

    args = demo_extract_3d_positions.parse_args()

    assert args.vlm_model_path == "/data/pretrain_models/Qwen/models--Qwen--Qwen2.5-VL-7B-Instruct"
    assert args.use_vlm_refinement is True
    assert args.mask_fallback == "auto"


def test_cli_mask_fallback_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo_extract_3d_positions.py", "--mask_fallback", "off"])

    args = demo_extract_3d_positions.parse_args()

    assert args.mask_fallback == "off"


def test_qwen_model_class_falls_back_to_auto_model(monkeypatch):
    class FakeTransformers:
        AutoModelForVision2Seq = object

    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)

    assert demo_extract_3d_positions._resolve_qwen_vl_model_class() is object



def test_cli_sample_index_range(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "demo_extract_3d_positions.py",
        "--sample_start_index",
        "400",
        "--sample_end_index",
        "500",
    ])

    args = demo_extract_3d_positions.parse_args()

    assert args.sample_start_index == 400
    assert args.sample_end_index == 500


def test_load_samples_selects_inclusive_index_range(tmpdir):
    dataset_path = tmpdir.join("annotations.json")
    samples = [
        {"question_index": idx, "question": "Where is the chair?", "answer": "yes", "image_filename": "x.jpg"}
        for idx in range(6)
    ]
    dataset_path.write('{"questions": ' + json.dumps(samples) + '}')

    args = argparse.Namespace(
        sample_json=None,
        jsonl=None,
        dataset_json=str(dataset_path),
        sample_id=None,
        sample_index=None,
        sample_start_index=2,
        sample_end_index=4,
        max_samples=None,
    )

    selected = demo_extract_3d_positions.load_samples(args)

    assert [sample["question_index"] for sample in selected] == [2, 3, 4]


def test_openai_refinement_model_uses_base_url(monkeypatch):
    calls = {}

    class FakeResponses:
        def create(self, **kwargs):
            class Response:
                output_text = "0"
            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            self.responses = FakeResponses()

    class FakeOpenAIModule:
        OpenAI = FakeOpenAI

    monkeypatch.setitem(sys.modules, "openai", FakeOpenAIModule)

    model = demo_extract_3d_positions.OpenAIVLRefinementModel(
        api_key="test-key",
        model="gpt-4.1",
        base_url="https://closeai.example/v1",
    )

    assert calls == {"api_key": "test-key", "base_url": "https://closeai.example/v1"}
    assert model.model == "gpt-4.1"


def test_object_extraction_prompt_has_question_type_rules():
    from object_3d_extraction.prompts import PROMPT_GET_OBJECTS_OF_INTEREST, QUESTION_TYPE_RULES

    assert "{question_type_rules}" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert set(["numeric_ct", "numeric_other", "yes_no", "choice_object"]).issubset(QUESTION_TYPE_RULES)
    assert "Question type: numeric count" in QUESTION_TYPE_RULES["numeric_ct"]
    assert "all visible instances" in QUESTION_TYPE_RULES["numeric_ct"]
    assert "do not invent indexed names" in QUESTION_TYPE_RULES["numeric_ct"]
    assert "handle_1" in QUESTION_TYPE_RULES["numeric_ct"]
    assert "[Detect] [handles, cabinets]" in QUESTION_TYPE_RULES["numeric_ct"]
    assert "Question type: numeric measurement or ratio" in QUESTION_TYPE_RULES["numeric_other"]
    assert "combined height" in QUESTION_TYPE_RULES["numeric_other"]
    assert "fit, stack, reach, or match" in QUESTION_TYPE_RULES["numeric_other"]
    assert "[Detect] [bedside tables, bed]" in QUESTION_TYPE_RULES["numeric_other"]
    assert "[Detect] [TV, sofa]" in QUESTION_TYPE_RULES["numeric_other"]
    assert "Question type: yes/no" in QUESTION_TYPE_RULES["yes_no"]
    assert "visibility" in QUESTION_TYPE_RULES["yes_no"]
    assert "Question type: object choice" in QUESTION_TYPE_RULES["choice_object"]
    assert "every physical object in the options must be included" in QUESTION_TYPE_RULES["choice_object"]


def test_object_extraction_prompt_documents_attribute_distinction_rule():
    from object_3d_extraction.prompts import PROMPT_GET_OBJECTS_OF_INTEREST, PROMPT_GET_OBJECTS_OF_INTEREST_AUX

    assert "Attribute distinction rule" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "gray chair" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "black chair" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [gray chair, table, black chair]" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "translucent cube" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "Do not merge same-category attributed objects" in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
    assert "Keep transparency phrases" in PROMPT_GET_OBJECTS_OF_INTEREST_AUX


def test_object_extraction_prompt_documents_camera_viewpoint_rule():
    from object_3d_extraction.prompts import PROMPT_GET_OBJECTS_OF_INTEREST

    assert "Camera rule" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert '"camera" usually means the viewpoint of the current image' in PROMPT_GET_OBJECTS_OF_INTEREST
    assert 'Do not include "camera" in [Detect]' in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "visible physical camera object" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "From the camera's perspective, is the chair on the left or right of the table?" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [chair, table]" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[camera, chair, table]" not in PROMPT_GET_OBJECTS_OF_INTEREST


def test_object_extraction_prompt_preserves_similar_object_categories():
    from object_3d_extraction.prompts import PROMPT_GET_OBJECTS_OF_INTEREST

    assert "Similar object rule" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert 'Do not replace "stool" with "chair"' in PROMPT_GET_OBJECTS_OF_INTEREST
    assert 'Do not replace "chair" with "stool"' in PROMPT_GET_OBJECTS_OF_INTEREST
    assert 'Do not replace "bench", "sofa", "couch", "ottoman", or "seat" with "chair"' in PROMPT_GET_OBJECTS_OF_INTEREST
    assert 'If the question says a generic "seat", keep "seat" as the detection target' in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Question] Is the chair to the left or right of the stool?" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [chair, stool]" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Question] Is the stool closer to the table than the chair?" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [stool, table, chair]" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [bench, chair]" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [sofa, chair]" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [ottoman, couch]" in PROMPT_GET_OBJECTS_OF_INTEREST
    assert "[Detect] [black chair, wooden stool]" in PROMPT_GET_OBJECTS_OF_INTEREST


def test_object_extraction_aux_prompt_preserves_similar_object_categories():
    from object_3d_extraction.prompts import PROMPT_GET_OBJECTS_OF_INTEREST_AUX

    assert "Return only one bracketed Python-style list" in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
    assert "Keep the exact object category used in the question" in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
    assert 'Keep full object phrases such as "black chair", "wooden stool", "white coffee table", or "person wearing a hat"' in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
    assert 'Do not replace "stool" with "chair"' in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
    assert 'Do not replace "chair" with "stool"' in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
    assert "Do not include relation words such as left, right, closer, farther, above, below, front, behind" in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
    assert 'Do not include "camera" unless it is a visible physical camera object' in PROMPT_GET_OBJECTS_OF_INTEREST_AUX
