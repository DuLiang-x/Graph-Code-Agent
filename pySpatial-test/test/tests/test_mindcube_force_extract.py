import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import mindcube


def test_mindcube_force_extract_cli_is_passed_to_sequential_processing(monkeypatch, tmpdir):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    def fake_load_entries(dataset_json, image_root, max_entries=None):
        return [{"id": "omni3d_0", "question": "q", "images": ["image.jpg"], "answer": "a"}]

    def fake_process(entry, agent, **kwargs):
        captured.update(kwargs)
        return {
            "scene_id": entry["id"],
            "scene_type": "UNKNOWN",
            "parse_success": False,
            "execution_success": False,
            "answer_generation_success": False,
            "answer_correct": False,
            "force_extract": kwargs.get("force_extract"),
        }

    monkeypatch.setattr(sys, "argv", [
        "mindcube.py",
        "--dataset_json", str(tmpdir.join("annotations.json")),
        "--mode", "graph",
        "--force_extract",
        "--max_code_repair_attempts", "2",
        "--output_file", str(tmpdir.join("out")),
        "--num_processes", "1",
    ])
    tmpdir.join("annotations.json").write("[]")
    monkeypatch.setattr(mindcube, "Agent", FakeAgent)
    monkeypatch.setattr(mindcube, "load_omni3d_entries", fake_load_entries)
    monkeypatch.setattr(mindcube, "process_scene_with_agent", fake_process)
    monkeypatch.setattr(mindcube, "create_sample_flowchart", lambda *args, **kwargs: None)

    mindcube.main()

    assert captured["force_extract"] is True
    assert captured["max_code_repair_attempts"] == 2


def test_mindcube_wrapper_passes_force_extract_to_worker(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    def fake_process(entry, agent, **kwargs):
        captured.update(kwargs)
        return {"scene_id": entry["id"]}

    monkeypatch.setattr(mindcube, "Agent", FakeAgent)
    monkeypatch.setattr(mindcube, "process_scene_with_agent", fake_process)

    mindcube.process_scene_with_agent_wrapper((
        {"id": "omni3d_0", "question": "q", "images": ["image.jpg"]},
        {"mode": "graph", "force_extract": True, "max_code_repair_attempts": 3},
    ))

    assert captured["force_extract"] is True
    assert captured["max_code_repair_attempts"] == 3


def test_mindcube_sequential_reuses_one_extractor(monkeypatch, tmpdir):
    built = []
    seen_extractors = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    def fake_build_object_extractor(**kwargs):
        extractor = object()
        built.append(extractor)
        return extractor

    def fake_load_entries(dataset_json, image_root, max_entries=None):
        return [
            {"id": "omni3d_0", "question": "q0", "images": ["image0.jpg"], "answer": "a"},
            {"id": "omni3d_1", "question": "q1", "images": ["image1.jpg"], "answer": "a"},
        ]

    def fake_process(entry, agent, **kwargs):
        seen_extractors.append(kwargs.get("extractor"))
        return {
            "scene_id": entry["id"],
            "scene_type": "UNKNOWN",
            "parse_success": False,
            "execution_success": False,
            "answer_generation_success": False,
            "answer_correct": False,
        }

    monkeypatch.setattr(sys, "argv", [
        "mindcube.py",
        "--dataset_json", str(tmpdir.join("annotations.json")),
        "--mode", "graph",
        "--output_file", str(tmpdir.join("out")),
        "--num_processes", "1",
    ])
    tmpdir.join("annotations.json").write("[]")
    monkeypatch.setattr(mindcube, "Agent", FakeAgent)
    monkeypatch.setattr(mindcube, "build_object_extractor", fake_build_object_extractor)
    monkeypatch.setattr(mindcube, "load_omni3d_entries", fake_load_entries)
    monkeypatch.setattr(mindcube, "process_scene_with_agent", fake_process)
    monkeypatch.setattr(mindcube, "create_sample_flowchart", lambda *args, **kwargs: None)

    mindcube.main()

    assert len(built) == 1
    assert seen_extractors == [built[0], built[0]]


def test_mindcube_worker_reuses_cached_extractor(monkeypatch):
    built = []
    seen_extractors = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    def fake_build_object_extractor(**kwargs):
        extractor = object()
        built.append(extractor)
        return extractor

    def fake_process(entry, agent, **kwargs):
        seen_extractors.append(kwargs.get("extractor"))
        return {"scene_id": entry["id"]}

    mindcube._WORKER_OBJECT_EXTRACTORS.clear()
    monkeypatch.setattr(mindcube, "Agent", FakeAgent)
    monkeypatch.setattr(mindcube, "build_object_extractor", fake_build_object_extractor)
    monkeypatch.setattr(mindcube, "process_scene_with_agent", fake_process)

    config = {"mode": "graph", "force_extract": True, "backend": "openai", "api_model": "gpt-4.1"}
    mindcube.process_scene_with_agent_wrapper(({"id": "omni3d_0", "question": "q", "images": []}, config))
    mindcube.process_scene_with_agent_wrapper(({"id": "omni3d_1", "question": "q", "images": []}, config))

    assert len(built) == 1
    assert seen_extractors == [built[0], built[0]]
