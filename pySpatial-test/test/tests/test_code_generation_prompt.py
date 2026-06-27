import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))
from agent.codeAgent.query import _available_graph_objects_prompt
from agent.prompt.template import (
    ANSWER_FORMAT_RULES,
    answer_prompt,
    api_specification,
    code_generation_prompt,
    code_repair_prompt,
)
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
        "handle_1": _box([3.0, 0.0, -3.0]),
        "handle_2": _box([4.0, 0.0, -3.0]),
    })

    prompt = _available_graph_objects_prompt(scene)

    assert "Available graph object names" in prompt
    assert "- fireplace" in prompt
    assert "- coffee table" in prompt
    assert "- leftmost cabinet" in prompt
    assert "- handle_1" in prompt
    assert "- handle_2" in prompt
    assert "Do not shorten names" in prompt
    assert "do not rewrite names into snake_case" in prompt
    assert "same-category instances may appear as indexed nodes" in prompt
    assert "Count these indexed nodes with graph.list_nodes()" in prompt


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


def test_graph_api_prompt_documents_indexed_counting_nodes():
    assert "same-category instances can be exposed as indexed nodes" in api_specification
    assert "handle_1, handle_2" in api_specification
    assert "Count indexed instance nodes from graph.list_nodes()" in api_specification
    assert 'For "how many X" visual counting questions' in api_specification
    assert 'graph.nodes_with_prefix("handle_")' in api_specification
    assert 'graph.count_prefix("handle_")' in api_specification
    assert "Do not answer counting or same-category multi-instance questions" in api_specification
    assert "count-ratio questions" in api_specification
    assert "sink_1 and sink_2" in api_specification
    assert "use graph.nodes_with_prefix(prefix) and graph.count_prefix(prefix)" in code_generation_prompt
    assert "For count-ratio questions" in code_generation_prompt
    assert "sink_1 and sink_2" in code_generation_prompt
    assert "Do not use a single aggregate node such as handles" in code_generation_prompt


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


def test_answer_prompt_includes_final_format_rules():
    assert "Final answer formatting rules" in ANSWER_FORMAT_RULES
    assert "computed_results" in ANSWER_FORMAT_RULES
    assert 'answer exactly "yes" or "no"' in ANSWER_FORMAT_RULES
    assert "answer with a single numeric value" in ANSWER_FORMAT_RULES
    assert "Do not output Python code" in ANSWER_FORMAT_RULES
    assert "Final answer formatting rules" in answer_prompt


def test_graph_api_prompt_documents_observer_selection_rules():
    assert "Observer selection rules for Omni3D-Bench" in api_specification
    assert "Always decide the observer before calling left/right/front/back APIs" in api_specification
    assert "For single-image Omni3D-Bench questions, use graph.observer_from_camera() by default" in api_specification
    assert 'Use graph.observer_from_object("object_name")' in api_specification
    assert 'Use graph.observer_from_to("from_object", "to_object")' in api_specification
    assert "do not call graph.is_left_of, graph.is_right_of, graph.is_in_front_of, or graph.is_behind without an observer" in api_specification


def test_code_generation_prompt_reminds_observer_choice():
    assert "Before writing code for left/right/front/back relations, first choose the observer" in code_generation_prompt
    assert "default to graph.observer_from_camera()" in code_generation_prompt
    assert "explicit object perspective or from-to perspective" in code_generation_prompt


def test_code_repair_prompt_documents_common_execution_fixes():
    assert "The previously generated code failed during execution" in code_repair_prompt
    assert "Keep the function signature as: def program(input_scene: Scene):" in code_repair_prompt
    assert "object name mismatch" in code_repair_prompt
    assert "graph.list_nodes()" in code_repair_prompt
    assert "Do not rewrite object names into snake_case" in code_repair_prompt
    assert "graph.closest_object" in code_repair_prompt
    assert "subscripting a float result" in code_repair_prompt
    assert "do not call pySpatial.extract_objects again" in code_repair_prompt


def test_graph_examples_include_omni3d_common_patterns():
    from agent.prompt.template import example_problems

    assert "Example 6: above/below relation" in example_problems
    assert 'graph.is_above("lamp", "table")' in example_problems
    assert 'graph.is_in_front_of("chair", "desk", camera)' in example_problems
    assert 'graph.distance("chair", "table")' in example_problems
    assert 'chair_h = graph.height("chair")' in example_problems
    assert 'observer = graph.observer_from_object("car")' in example_problems
    assert 'observer = graph.observer_from_to("person", "tv")' in example_problems
    assert "use the real node names that exist in graph.list_nodes()" in example_problems


def test_camera_semantics_are_documented_for_graph_prompt():
    from agent.prompt.template import example_problems

    assert "Camera semantics" in api_specification
    assert "Camera is the coordinate origin [0, 0, 0] / image viewpoint" in api_specification
    assert 'Do not detect, segment, or localize "camera" as an object' in api_specification
    assert 'Do not expect "camera" to appear in graph.list_nodes()' in api_specification
    assert 'Do not call graph.observer_from_object("camera")' in api_specification
    assert 'Never use graph.observer_from_object("camera") for camera perspective' in api_specification
    assert "the question says \"from the camera's perspective\"" in api_specification
    assert 'the question says "from the image perspective"' in api_specification
    assert "the question says \"from the viewer's perspective\"" in api_specification
    assert "visible object's own perspective" in api_specification
    assert 'Do not call graph.observer_from_object("camera").' in code_generation_prompt
    assert "Do not treat camera as a graph node" in code_generation_prompt
    assert "Example 12: camera perspective is not a graph node" in example_problems
    assert 'camera = graph.observer_from_camera()' in example_problems
    assert 'graph.is_left_of("chair", "table", camera)' in example_problems
    assert 'Do not use graph.observer_from_object("camera"). The camera is the current image viewpoint, not an object node.' in example_problems


def test_graph_examples_include_indexed_counting_pattern():
    from agent.prompt.template import example_problems

    assert "Example 13: counting indexed same-category instances" in example_problems
    assert 'handles = graph.nodes_with_prefix("handle_")' in example_problems
    assert 'answer = graph.count_prefix("handle_")' in example_problems
    assert '"counted_nodes": handles' in example_problems

def test_graph_api_prompt_documents_minimal_geometry_helpers():
    assert "graph.distance_to_camera(obj) -> float" in api_specification
    assert "graph.closest_to_camera(candidates=None) -> str | None" in api_specification
    assert "graph.furthest_from_camera(candidates=None) -> str | None" in api_specification
    assert "graph.observer_from_object_to_camera(obj) -> Observer" in api_specification
    assert "graph.volume(obj) -> float" in api_specification
    assert "graph.screen_diagonal(obj) -> float" in api_specification
    assert "graph.nodes_with_prefix(prefix) -> list[str]" in api_specification
    assert "graph.count_prefix(prefix) -> int" in api_specification
    assert "graph.footprint_overlap(a, b" in api_specification
    assert "graph.vertical_clearance(upper, lower) -> float" in api_specification
    assert "graph.would_collide_along_direction" in api_specification
    assert "use graph.distance_to_camera(obj), graph.closest_to_camera(candidates), or graph.furthest_from_camera(candidates)" in api_specification
    assert "use graph.observer_from_object_to_camera(X)" in api_specification
    assert "use graph.volume(obj)" in api_specification
    assert "use graph.screen_diagonal(obj)" in code_generation_prompt
    assert "graph.distance_to_camera/closest_to_camera/furthest_from_camera" in code_generation_prompt
    assert "graph.would_collide_along_direction or graph.footprint_overlap" in code_generation_prompt


def test_graph_examples_use_minimal_geometry_helpers():
    from agent.prompt.template import example_problems

    assert "graph.screen_diagonal(tv)" in example_problems
    assert 'graph.volume("bedside table")' in example_problems
    assert "graph.closest_to_camera(candidates)" in example_problems
    assert "graph.would_collide_along_direction" in example_problems
    assert 'graph.count_prefix("handle_")' in example_problems

def test_codeagent_prompt_documents_badcase_semantic_guards():
    assert 'never call graph.distance(obj, "camera")' in code_generation_prompt
    assert 'Do not call graph.observer_from_object("camera")' in code_generation_prompt
    assert "do not use //, int(), round(), floor(), or ceil()" in code_generation_prompt
    assert "use graph.screen_diagonal(obj); do not include depth" in code_generation_prompt
    assert "return a needs_visual_* computed result" in code_generation_prompt
    assert "do not decide from a single z/y threshold" in code_generation_prompt
    assert "Do not use //, int(), round(), floor(), or ceil()" in api_specification
    assert "does not include depth" in api_specification
    assert "Return needs_visual_visibility_check" in api_specification
    assert "needs_visual_clock_reading" in api_specification


def test_codeagent_prompt_documents_plural_not_automatic_counting():
    assert "Plural words alone do not automatically mean counting" in code_generation_prompt
    assert "compute a continuous size ratio instead of counting visible instances" in code_generation_prompt
    assert "coordinate origin [0, 0, 0]" in code_generation_prompt
    assert "Plural object words alone do not mean the task is counting" in api_specification
    assert "How many of X would reach the height of Y" in api_specification


def test_codeagent_examples_include_stack_ratio_and_closer_to_a_or_b():
    from agent.prompt.template import example_problems

    assert "Example 13a: count-ratio with attributed same-category instances" in example_problems
    assert 'graph.count_prefix("brown_chair_")' in example_problems
    assert 'graph.count_prefix("black_chair_")' in example_problems
    assert "Example 13b: stack/reach height is a continuous ratio, not counting" in example_problems
    assert 'graph.height("rightmost stool")' in example_problems
    assert 'graph.height("leftmost chair")' in example_problems
    assert "height ratio, not visual counting" in example_problems
    assert "Example 13c: choose whether X is closer to A or B" in example_problems
    assert 'graph.distance("rightmost chair", "table")' in example_problems
    assert 'graph.distance("rightmost chair", "wooden dresser")' in example_problems


def test_answer_prompt_preserves_numeric_and_visual_fallback_format():
    assert "keep the numeric value and do not round it to 0" in ANSWER_FORMAT_RULES
    assert "needs_visual_*" in ANSWER_FORMAT_RULES
    assert 'answer exactly "yes" or "no"' in ANSWER_FORMAT_RULES
    assert "answer with a single numeric value" in ANSWER_FORMAT_RULES

