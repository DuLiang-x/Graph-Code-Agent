import json
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from pySpatial_Interface import OBJECT_OUTPUT_ROOTS, Scene, pySpatial


def sample_result():
    return {
        "chair": {
            "position": [0.0, 0.0, -2.0],
            "box3d_center": [0.0, 0.0, -2.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, -0.5, -2.5],
            "box3d_max": [0.5, 0.5, -1.5],
            "orientation": [0.0, 0.0, -1.0],
        }
    }


def test_extract_objects_reads_cached_json(monkeypatch, tmpdir):
    scene_id = "scene_cached"
    output_root = Path(str(tmpdir)) / "Omni3D-Bench"
    sample_dir = output_root / scene_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "object_3d_positions.json").write_text(
        json.dumps({scene_id: {"result": sample_result()}})
    )
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "auto", output_root)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    result = pySpatial.extract_objects(scene, mask_fallback="auto")

    assert result == sample_result()
    assert scene.object_3d_boxes == sample_result()


def test_extract_objects_calls_demo_wrapper_when_json_missing(monkeypatch, tmpdir):
    scene_id = "scene_missing"
    output_root = Path(str(tmpdir)) / "Omni3D-Benchnomask"
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "off", output_root)

    import scripts.demo_extract_3d_positions as demo

    calls = {}

    def fake_extract(sample, output_root, **kwargs):
        calls["sample"] = sample
        calls["output_root"] = Path(output_root)
        calls["mask_fallback"] = kwargs["mask_fallback"]
        return {"result": sample_result()}, str(Path(output_root) / sample["id"] / "object_3d_positions.json")

    monkeypatch.setattr(demo, "extract_objects_for_sample", fake_extract)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    result = pySpatial.extract_objects(scene, mask_fallback="off", use_vlm_refinement=False)

    assert result == sample_result()
    assert calls["sample"]["id"] == scene_id
    assert calls["output_root"] == output_root
    assert calls["mask_fallback"] == "off"


def test_build_graph_from_scene_boxes():
    scene = Scene(["image.jpg"], "Where is the chair?", scene_id="scene_graph")
    scene.object_3d_boxes = sample_result()

    graph = pySpatial.build_graph(scene)

    assert graph.list_nodes() == ["chair"]
    assert scene.spatial_graph is graph


def test_extract_objects_reads_legacy_nomask_json(monkeypatch, tmpdir):
    from pySpatial_Interface import LEGACY_OBJECT_OUTPUT_ROOTS

    scene_id = "scene_legacy"
    primary_root = Path(str(tmpdir)) / "Omni3D-Benchnomask"
    legacy_root = Path(str(tmpdir)) / "Omni3D-Bench-nomask"
    sample_dir = legacy_root / scene_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "object_3d_positions.json").write_text(
        json.dumps({scene_id: {"result": sample_result()}})
    )
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "off", primary_root)
    monkeypatch.setitem(LEGACY_OBJECT_OUTPUT_ROOTS, "off", legacy_root)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    result = pySpatial.extract_objects(scene, mask_fallback="off")

    assert result == sample_result()


def test_extract_objects_passes_unified_openai_model_config(monkeypatch, tmpdir):
    scene_id = "scene_openai"
    output_root = Path(str(tmpdir)) / "Omni3D-Bench"
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "auto", output_root)

    import scripts.demo_extract_3d_positions as demo

    calls = {}

    def fake_extract(sample, output_root, **kwargs):
        calls.update(kwargs)
        return {"result": sample_result()}, str(Path(output_root) / sample["id"] / "object_3d_positions.json")

    monkeypatch.setattr(demo, "extract_objects_for_sample", fake_extract)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    pySpatial.extract_objects(
        scene,
        mask_fallback="auto",
        backend="openai",
        api_model="gpt-4.1",
        api_key="test-key",
    )

    assert calls["backend"] == "openai"
    assert calls["api_model"] == "gpt-4.1"
    assert calls["api_key"] == "test-key"


def test_extract_objects_passes_base_url_to_demo_wrapper(monkeypatch, tmpdir):
    scene_id = "scene_base_url"
    output_root = Path(str(tmpdir)) / "Omni3D-Bench"
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "auto", output_root)

    import scripts.demo_extract_3d_positions as demo

    calls = {}

    def fake_extract(sample, output_root, **kwargs):
        calls.update(kwargs)
        return {"result": sample_result()}, str(Path(output_root) / sample["id"] / "object_3d_positions.json")

    monkeypatch.setattr(demo, "extract_objects_for_sample", fake_extract)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    pySpatial.extract_objects(
        scene,
        mask_fallback="auto",
        backend="openai",
        api_model="gpt-4.1",
        api_key="test-key",
        base_url="https://closeai.example/v1",
    )

    assert calls["base_url"] == "https://closeai.example/v1"


def test_resolve_openai_base_url_prefers_closeai_env(monkeypatch):
    from pySpatial_Interface import resolve_openai_base_url

    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-compatible.example/v1")
    monkeypatch.setenv("CLOSEAI_BASE_URL", "https://closeai.example/v1")

    assert resolve_openai_base_url() == "https://closeai.example/v1"
    assert resolve_openai_base_url("https://explicit.example/v1") == "https://explicit.example/v1"

def test_extract_objects_reads_save_dir_cache_before_default_outputs(monkeypatch, tmpdir):
    scene_id = "scene_custom_cache"
    default_root = Path(str(tmpdir)) / "default"
    custom_root = Path(str(tmpdir)) / "custom"
    sample_dir = custom_root / scene_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "object_3d_positions.json").write_text(
        json.dumps({scene_id: {"result": sample_result()}})
    )
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "auto", default_root)

    import scripts.demo_extract_3d_positions as demo

    def fail_extract(*args, **kwargs):
        raise AssertionError("extract wrapper should not be called when save_dir cache exists")

    monkeypatch.setattr(demo, "extract_objects_for_sample", fail_extract)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    result = pySpatial.extract_objects(scene, mask_fallback="auto", save_dir=str(custom_root))

    assert result == sample_result()


def test_extract_objects_writes_missing_save_dir_cache_to_custom_root(monkeypatch, tmpdir):
    scene_id = "scene_custom_missing"
    default_root = Path(str(tmpdir)) / "default"
    custom_root = Path(str(tmpdir)) / "custom"
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "auto", default_root)

    import scripts.demo_extract_3d_positions as demo

    calls = {}

    def fake_extract(sample, output_root, **kwargs):
        calls["sample"] = sample
        calls["output_root"] = Path(output_root)
        return {"result": sample_result()}, str(Path(output_root) / sample["id"] / "object_3d_positions.json")

    monkeypatch.setattr(demo, "extract_objects_for_sample", fake_extract)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    result = pySpatial.extract_objects(scene, mask_fallback="auto", save_dir=str(custom_root))

    assert result == sample_result()
    assert calls["sample"]["id"] == scene_id
    assert calls["output_root"] == custom_root



def test_extract_objects_force_extract_skips_existing_cache(monkeypatch, tmpdir):
    scene_id = "scene_force_extract"
    default_root = Path(str(tmpdir)) / "default"
    custom_root = Path(str(tmpdir)) / "custom"
    sample_dir = custom_root / scene_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "object_3d_positions.json").write_text(
        json.dumps({scene_id: {"result": {"stale": {"position": [1, 2, 3]}}}})
    )
    monkeypatch.setitem(OBJECT_OUTPUT_ROOTS, "auto", default_root)

    import scripts.demo_extract_3d_positions as demo

    calls = {}

    def fake_extract(sample, output_root, **kwargs):
        calls["sample"] = sample
        calls["output_root"] = Path(output_root)
        return {"result": sample_result()}, str(Path(output_root) / sample["id"] / "object_3d_positions.json")

    monkeypatch.setattr(demo, "extract_objects_for_sample", fake_extract)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    result = pySpatial.extract_objects(
        scene,
        mask_fallback="auto",
        save_dir=str(custom_root),
        force_extract=True,
    )

    assert result == sample_result()
    assert calls["sample"]["id"] == scene_id
    assert calls["output_root"] == custom_root


def test_extract_objects_default_cache_still_skips_wrapper(monkeypatch, tmpdir):
    scene_id = "scene_force_extract_off"
    custom_root = Path(str(tmpdir)) / "custom"
    sample_dir = custom_root / scene_id
    sample_dir.mkdir(parents=True)
    (sample_dir / "object_3d_positions.json").write_text(
        json.dumps({scene_id: {"result": sample_result()}})
    )

    import scripts.demo_extract_3d_positions as demo

    def fail_extract(*args, **kwargs):
        raise AssertionError("extract wrapper should not be called without force_extract")

    monkeypatch.setattr(demo, "extract_objects_for_sample", fail_extract)

    scene = Scene(["image.jpg"], "Where is the chair?", scene_id=scene_id)
    result = pySpatial.extract_objects(scene, mask_fallback="auto", save_dir=str(custom_root))

    assert result == sample_result()
