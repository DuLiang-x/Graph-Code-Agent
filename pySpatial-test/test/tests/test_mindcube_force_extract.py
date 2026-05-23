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
        {"mode": "graph", "force_extract": True},
    ))

    assert captured["force_extract"] is True
