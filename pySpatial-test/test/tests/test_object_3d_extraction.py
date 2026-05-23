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
from object_3d_extraction.object_3d_locator import detection_prompts_for_object, maybe_fallback_mask, parse_candidate_index, rank_detection_candidates
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
        assert demo_extract_3d_positions.resolve_object_names(sample) == expected


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
    assert detection_prompts_for_object("tv") == ["tv", "television"]

def test_rule_ranker_avoids_large_area_box_for_non_area_object():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [0, 40, 100, 100], "score": 0.9, "prompt": "white coffee table"},
        {"box2d": [10, 50, 45, 75], "score": 0.55, "prompt": "table"},
    ]

    selected = rank_detection_candidates(image, "white coffee table", candidates, {})

    assert selected["box2d"] == [10, 50, 45, 75]
    assert selected["prompt"] == "table"


def test_rule_ranker_selects_rightmost_candidate():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [5, 10, 25, 40], "score": 0.9, "prompt": "sofa"},
        {"box2d": [70, 10, 95, 40], "score": 0.5, "prompt": "sofa"},
    ]

    selected = rank_detection_candidates(image, "rightmost sofa", candidates, {})

    assert selected["box2d"] == [70, 10, 95, 40]


def test_entity_ranker_prefers_tight_white_coffee_table_over_large_carpet_box():
    image = Image.new("RGB", (100, 100), color="white")
    candidates = [
        {"box2d": [0, 35, 100, 100], "score": 0.9, "prompt": "white coffee table"},
        {"box2d": [10, 45, 45, 72], "score": 0.58, "prompt": "white table"},
    ]

    selected = rank_detection_candidates(image, "white coffee table", candidates, {})

    assert selected["box2d"] == [10, 45, 45, 72]
    assert "large_entity_penalty" not in selected["rank_reasons"]


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
    def process_messages(self, messages, max_new_tokens=32):
        assert messages[0]["content"][0]["type"] == "image"
        return "1"


def test_vlm_refinement_selects_mocked_candidate_index():
    locator = make_locator(
        {
            "chair": [
                {"box2d": [1, 1, 5, 5], "score": 0.9},
                {"box2d": [8, 8, 12, 12], "score": 0.75},
            ]
        },
        vlm_model=FakeVLM(),
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


def test_vlm_refinement_rejects_small_unclear_black_table():
    locator = make_locator(
        {
            "black table": [
                {"box2d": [27, 72, 71, 98], "score": 0.47, "prompt": "black table"},
                {"box2d": [38, 35, 48, 45], "score": 0.20, "prompt": "black table"},
            ]
        },
        vlm_model=FakeVLM(),
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (100, 100), color="white")

    result = locator.extract(image, ["black table"])

    item = result["black table"]
    assert item["box2d"] == [27, 72, 71, 98]
    assert item["candidate_rank_reason"] == "rule_ranker"
    assert item["rule_selected_index"] == 0
    assert item["vlm_selected_index"] == 1
    assert item["final_selected_index"] == 0
    assert item["selection_decision"] == "rule_ranker_vlm_rejected"
    assert item["selection_reject_reason"] in {"rank_score_too_low", "partial_entity_box"}


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
