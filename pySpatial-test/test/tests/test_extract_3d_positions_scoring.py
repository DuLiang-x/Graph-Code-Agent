import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_TEST_DIR = Path(__file__).resolve().parents[1]
if str(REPO_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_TEST_DIR))

from object_3d_extraction import Object3DExtractionConfig
from scripts import extract_3d_positions
from scripts.extract_3d_positions import ScoringObject3DLocator, build_scoring_config


def test_extract_script_is_self_contained_from_demo_and_scoring_scripts():
    source = (REPO_TEST_DIR / "scripts" / "extract_3d_positions.py").read_text()

    assert "demo_extract_3d_positions" not in source
    assert "test_vlm_candidate_scoring" not in source


class FakeDetection:
    def detect(self, image, prompt, max_candidates=None):
        return [
            {"box2d": [10, 10, 50, 50], "score": 0.9},
            {"box2d": [60, 10, 110, 60], "score": 0.8},
        ]

    def run_segmentation(self, image, box):
        mask = np.zeros((image.size[1], image.size[0]), dtype=float)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 1.0
        return mask


class FakeDepth:
    def run_depth_estimation(self, image):
        return np.ones((image.size[1], image.size[0]), dtype=float)

    def unproject_to_3D(self, image, depth, mask):
        return {
            "position": [0, 0, 1],
            "box3d_size": [1, 1, 1],
            "box3d_center": [0, 0, 1],
            "box3d_min": [-0.5, -0.5, 0.5],
            "box3d_max": [0.5, 0.5, 1.5],
            "mask_area": int(mask.sum()),
            "depth_mode": 1.0,
            "num_points": int(mask.sum()),
        }


class FakeOrientation:
    def run_orientation_estimation(self, image, box):
        return {"orientation": [0, 0, 0], "orientation_angles": [0, 0, 0]}


class FakeVLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def process_messages(self, messages, max_new_tokens=768):
        self.calls += 1
        return self.response


def make_locator(vlm):
    config = Object3DExtractionConfig()
    config.detection.use_vlm_refinement = True
    config.detection.count_max_instances = 20
    return ScoringObject3DLocator(
        config=config,
        device="cpu",
        detection_module=FakeDetection(),
        depth_module=FakeDepth(),
        orientation_module=FakeOrientation(),
        vlm_model=vlm,
    )


def test_scoring_locator_uses_vlm_selected_candidate_for_non_counting_object():
    vlm = FakeVLM('{"scores":[{"index":1,"score":95,"reason":"better"}],"selected_index":1}')
    locator = make_locator(vlm)
    image = Image.new("RGB", (128, 96), "white")

    result = locator.extract(image, ["chair"], question="Which chair?", visualize=False)

    item = result["chair"]
    assert item["selection_decision"] == "vlm_candidate_scoring"
    assert item["final_selected_index"] == 1
    assert item["box2d"] == [60, 10, 110, 60]
    assert item["vlm_candidate_scores"]["selected_index"] == 1
    assert vlm.calls == 1


def test_scoring_locator_keeps_counting_path_without_vlm_scoring():
    vlm = FakeVLM("this should not be used")
    locator = make_locator(vlm)
    image = Image.new("RGB", (128, 96), "white")

    result = locator.extract(
        image,
        ["handles"],
        question="How many handles are visible?",
        question_type="number_vt",
        visualize=False,
    )

    assert result["handles"]["counting_target"] is True
    assert result["handles"]["counting_instance_count"] == 2
    assert vlm.calls == 0


def test_build_scoring_config_uses_cli_thresholds_and_count_limit():
    args = type(
        "Args",
        (),
        {
            "box_threshold": 0.12,
            "text_threshold": 0.34,
            "use_vlm_refinement": False,
            "count_max_instances": 17,
            "mask_fallback": "off",
        },
    )()

    config = build_scoring_config(args)

    assert config.detection.box_threshold == 0.12
    assert config.detection.text_threshold == 0.34
    assert config.detection.use_vlm_refinement is False
    assert config.detection.count_max_instances == 17
    assert config.mask_fallback_mode == "off"


def test_main_writes_error_record_and_continues(monkeypatch, tmpdir, capsys):
    tmp_path = Path(str(tmpdir))
    samples = [
        {"question_index": 1, "question": "bad sample"},
        {"question_index": 2, "question": "good sample"},
    ]
    args = type(
        "Args",
        (),
        {
            "save_dir": str(tmp_path),
            "device": "cpu",
            "use_vlm_refinement": False,
            "use_vlm_object_extraction": False,
            "box_threshold": 0.05,
            "text_threshold": 0.05,
            "count_max_instances": 20,
            "mask_fallback": "auto",
            "max_new_tokens": 768,
            "no_visualize": True,
            "image": None,
            "base_data_path": None,
            "backend": "openai",
            "api_model": "gpt-4.1",
            "api_key": None,
            "base_url": None,
            "vlm_model_path": "unused",
        },
    )()

    class FakeLocator:
        def __init__(self, *args, **kwargs):
            pass

        def extract(self, **kwargs):
            return {"chair": {"box2d": [1, 2, 3, 4]}}

    def fake_resolve_image_path(args_obj, sample):
        if sample["question_index"] == 1:
            raise RuntimeError("missing image")
        return "/tmp/good.png"

    def fake_resolve_object_names(sample, image=None, vlm_model=None, use_vlm_object_extraction=True):
        return {
            "objects": ["chair"],
            "method": "rule",
            "vlm_extracted_objects": [],
            "rule_extracted_objects": ["chair"],
            "object_extraction_response": None,
            "question_type": "number_other",
        }

    monkeypatch.setattr(extract_3d_positions, "parse_args", lambda: args)
    monkeypatch.setattr(extract_3d_positions, "load_samples", lambda args_obj: samples)
    monkeypatch.setattr(extract_3d_positions, "build_vlm_refinement_model", lambda args_obj: None)
    monkeypatch.setattr(extract_3d_positions, "ScoringObject3DLocator", FakeLocator)
    monkeypatch.setattr(extract_3d_positions, "resolve_image_path", fake_resolve_image_path)
    monkeypatch.setattr(extract_3d_positions, "resolve_object_names", fake_resolve_object_names)

    extract_3d_positions.main()

    first = tmp_path / "omni3d_1" / "object_3d_positions.json"
    second = tmp_path / "omni3d_2" / "object_3d_positions.json"
    assert first.exists()
    assert second.exists()
    assert "missing image" in first.read_text()
    assert "vlm_candidate_scoring" in second.read_text()
    assert "written_files" in capsys.readouterr().out
