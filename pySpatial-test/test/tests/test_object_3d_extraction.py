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
from object_3d_extraction.object_3d_locator import detection_prompts_for_object, parse_candidate_index, rank_detection_candidates
from object_3d_extraction.utils import extract_object_names_from_question_options


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


def make_locator(detections, vlm_model=None, use_vlm_refinement=False):
    cfg = Object3DExtractionConfig(min_mask_area=1, min_depth_points=1)
    cfg.detection.use_vlm_refinement = use_vlm_refinement
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


class FakeVLM:
    def process_messages(self, messages, max_new_tokens=32):
        assert messages[0]["content"][0]["type"] == "image"
        return "1"


def test_vlm_refinement_selects_mocked_candidate_index():
    locator = make_locator(
        {
            "chair": [
                {"box2d": [1, 1, 4, 4], "score": 0.9},
                {"box2d": [8, 8, 14, 14], "score": 0.6},
            ]
        },
        vlm_model=FakeVLM(),
        use_vlm_refinement=True,
    )
    image = Image.new("RGB", (16, 16), color="white")

    result = locator.extract(image, ["chair"])

    item = result["chair"]
    assert item["box2d"] == [8, 8, 14, 14]
    assert item["candidate_rank_reason"] == "vlm_refinement"

def test_vlm_refinement_invalid_response_falls_back_to_first_candidate():
    assert parse_candidate_index("not an index", 3) == 0
    assert parse_candidate_index("9", 3) == 0


def test_cli_default_vlm_model_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo_extract_3d_positions.py"])

    args = demo_extract_3d_positions.parse_args()

    assert args.vlm_model_path == "/data/pretrain_models/Qwen/models--Qwen--Qwen2.5-VL-7B-Instruct"
    assert args.use_vlm_refinement is True

def test_qwen_model_class_falls_back_to_auto_model(monkeypatch):
    class FakeTransformers:
        AutoModelForVision2Seq = object

    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)

    assert demo_extract_3d_positions._resolve_qwen_vl_model_class() is object

