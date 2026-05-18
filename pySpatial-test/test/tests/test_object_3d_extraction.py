import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from object_3d_extraction import Object3DExtractionConfig, Object3DLocator
from object_3d_extraction.depth_module import unproject_to_3D
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


def make_locator(detections):
    cfg = Object3DExtractionConfig(min_mask_area=1, min_depth_points=1)
    return Object3DLocator(
        config=cfg,
        device="cpu",
        detection_module=FakeDetectionModule(detections),
        depth_module=FakeDepthModule(),
        orientation_module=FakeOrientationModule(),
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
