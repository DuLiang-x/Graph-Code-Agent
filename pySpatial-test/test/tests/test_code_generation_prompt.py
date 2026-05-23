import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))
from agent.codeAgent.query import _available_graph_objects_prompt
from agent.prompt.template import api_specification
from pySpatial_Interface import Scene
from spatial_graph import SpatialGraph


def _box(position):
    return {
        "position": position,
        "box3d_center": position,
        "box3d_size": [1.0, 1.0, 1.0],
        "box3d_min": [position[0] - 0.5, position[1] - 0.5, position[2] - 0.5],
        "box3d_max": [position[0] + 0.5, position[1] + 0.5, position[2] + 0.5],
        "orientation": [0.0, 0.0, -1.0],
    }


def test_available_graph_objects_prompt_lists_exact_names():
    scene = Scene([], "question", scene_id="scene")
    scene.spatial_graph = SpatialGraph({
        "fireplace": _box([0.0, 0.0, -3.0]),
        "coffee table": _box([1.0, 0.0, -3.0]),
        "leftmost cabinet": _box([2.0, 0.0, -3.0]),
    })

    prompt = _available_graph_objects_prompt(scene)

    assert "Available graph object names" in prompt
    assert "- fireplace" in prompt
    assert "- coffee table" in prompt
    assert "- leftmost cabinet" in prompt
    assert "Do not shorten names" in prompt
    assert "do not rewrite names into snake_case" in prompt


def test_graph_api_prompt_documents_closest_object_and_float_returns():
    assert "graph.closest_object(target, candidates=None)" in api_specification
    assert "does not accept an observer argument" in api_specification
    assert "Do not write graph.closest_object(..., observer=camera)" in api_specification
    assert "graph.distance(a, b) -> float" in api_specification
    assert "graph.size_ratio(a, b) -> float" in api_specification
    assert 'Wrong: graph.size_ratio("tv", "table")[0]' in api_specification

def test_graph_api_prompt_defaults_spatial_directions_to_camera_viewpoint():
    assert "camera/image viewpoint by default" in api_specification
    assert "Use graph.observer_from_camera() unless the question explicitly says" in api_specification
    examples = __import__("agent.prompt.template", fromlist=["example_problems"]).example_problems
    assert 'graph.is_right_of("sofa", "coffee table", camera)' in api_specification or "graph.is_right_of(target, reference, camera)" in examples

def test_graph_api_prompt_documents_dimension_apis_and_safe_ratio():
    assert "graph.height(obj) -> float" in api_specification
    assert "graph.width(obj) -> float" in api_specification
    assert "graph.depth(obj) -> float" in api_specification
    assert 'graph.length(obj, axis="auto") -> float' in api_specification
    assert "graph.ratio(numerator, denominator, eps=1e-9) -> float" in api_specification
    assert "signed size differences" in api_specification
    assert "must not be used as an object's own height" in api_specification
    assert 'For "height of X", use graph.height("X").' in api_specification
    assert "use graph.ratio(numerator, denominator) instead of direct division" in api_specification


def test_graph_examples_include_height_ratio_pattern():
    from agent.prompt.template import example_problems

    assert 'fireplace_h = graph.height("fireplace")' in example_problems
    assert 'table_h = graph.height("coffee table")' in example_problems
    assert 'ratio = graph.ratio(fireplace_h, table_h + sofa_h)' in example_problems

def test_prompt_documents_known_size_calibration_rule():
    from agent.prompt.template import example_problems

    assert "Raw graph dimensions are 3D AABB units" in api_specification
    assert 'If the question gives a known real size such as "X is 2m long"' in api_specification
    assert "scale = known_real_size / graph.length(reference_object)" in api_specification
    assert "Do not directly return raw graph.length(target_object) as meters" in api_specification
    assert "table_length_m = 2.0" in example_problems
    assert "answer = graph.ratio(sofa_raw * table_length_m, table_raw)" in example_problems

