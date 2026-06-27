import numpy as np
import pytest
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


from spatial_graph import SpatialGraph


def boxes():
    return {
        "left": {
            "position": [-1.0, 0.0, -3.0],
            "box3d_center": [-1.0, 0.0, -3.0],
            "box3d_size": [1.0, 2.0, 1.0],
            "box3d_min": [-1.5, -1.0, -3.5],
            "box3d_max": [-0.5, 1.0, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "right": {
            "position": [1.0, 1.0, -3.0],
            "box3d_center": [1.0, 1.0, -3.0],
            "box3d_size": [2.0, 2.0, 2.0],
            "box3d_min": [0.0, 0.0, -4.0],
            "box3d_max": [2.0, 2.0, -2.0],
            "orientation": [0.0, 0.0, -1.0],
        },
    }


def test_spatial_graph_basic_queries():
    graph = SpatialGraph(boxes())
    camera = graph.observer_from_camera()

    assert graph.list_nodes() == ["left", "right"]
    assert np.isclose(graph.distance("left", "right"), np.sqrt(5.0))
    assert graph.is_left_of("left", "right", camera)
    assert graph.is_right_of("right", "left", camera)
    assert graph.is_above("right", "left")
    assert graph.size_ratio("right", "left") == 4.0


def test_spatial_graph_with_state_restores_changes():
    graph = SpatialGraph(boxes())
    before = graph.get_node("left").position.copy()

    with graph.with_state():
        graph.move_object("left", [10.0, 0.0, 0.0])
        assert graph.get_node("left").position[0] == before[0] + 10.0

    assert np.allclose(graph.get_node("left").position, before)

def test_spatial_graph_get_node_accepts_lightweight_name_aliases():
    graph = SpatialGraph({
        "coffee table": {
            "position": [0.0, 0.0, -3.0],
            "box3d_center": [0.0, 0.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, -0.5, -3.5],
            "box3d_max": [0.5, 0.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        }
    })

    exact = graph.get_node("coffee table")

    assert graph.get_node("coffee_table") is exact
    assert graph.get_node("coffee-table") is exact
    assert graph.get_node("coffee   table") is exact
    assert graph.get_node("Coffee_Table") is exact

    tv_graph = SpatialGraph({
        "tv": {
            "position": [0.0, 0.0, -3.0],
            "box3d_center": [0.0, 0.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, -0.5, -3.5],
            "box3d_max": [0.5, 0.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        }
    })
    assert tv_graph.get_node("TV") is tv_graph.get_node("tv")


def test_spatial_graph_get_node_error_lists_available_nodes():
    graph = SpatialGraph(boxes())

    with pytest.raises(KeyError) as exc_info:
        graph.get_node("coffee_table")

    message = str(exc_info.value)
    assert "Unknown object 'coffee_table'" in message
    assert "Available objects: left, right" in message

def test_spatial_graph_get_node_does_not_fuzzy_match_short_names():
    graph = SpatialGraph({
        "leftmost cabinet": {
            "position": [0.0, 0.0, -3.0],
            "box3d_center": [0.0, 0.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, -0.5, -3.5],
            "box3d_max": [0.5, 0.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "center cabinet": {
            "position": [1.0, 0.0, -3.0],
            "box3d_center": [1.0, 0.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [0.5, -0.5, -3.5],
            "box3d_max": [1.5, 0.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        },
    })

    with pytest.raises(KeyError) as exc_info:
        graph.get_node("left")

    message = str(exc_info.value)
    assert "Unknown object 'left'" in message
    assert "Available objects: leftmost cabinet, center cabinet" in message

def test_spatial_graph_object_dimension_apis_and_safe_ratio():
    graph = SpatialGraph(boxes())

    assert graph.height("right") == 2.0
    assert graph.width("right") == 2.0
    assert graph.depth("right") == 2.0
    assert graph.length("left") == max(graph.width("left"), graph.depth("left"))
    assert graph.length("left", axis="width") == graph.width("left")
    assert graph.length("left", axis="depth") == graph.depth("left")
    assert graph.compare_width("left", "right") == -1.0
    assert graph.ratio(1.0, 0.0) == float("inf")
    assert graph.ratio(4.0, 2.0) == 2.0


def test_spatial_graph_length_rejects_unknown_axis():
    graph = SpatialGraph(boxes())

    with pytest.raises(ValueError):
        graph.length("left", axis="diagonal")

def test_spatial_graph_camera_helpers():
    graph = SpatialGraph({
        "chair_1": {
            "position": [0.0, 0.0, -2.0],
            "box3d_center": [0.0, 0.0, -2.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, -0.5, -2.5],
            "box3d_max": [0.5, 0.5, -1.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "chair_2": {
            "position": [0.0, 0.0, -5.0],
            "box3d_center": [0.0, 0.0, -5.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, -0.5, -5.5],
            "box3d_max": [0.5, 0.5, -4.5],
            "orientation": [0.0, 0.0, -1.0],
        },
    })

    assert graph.distance_to_camera("chair_1") == 2.0
    assert graph.closest_to_camera(["chair_1", "chair_2"]) == "chair_1"
    assert graph.furthest_from_camera(["chair_1", "chair_2"]) == "chair_2"
    observer = graph.observer_from_object_to_camera("chair_1")
    _, _, forward = observer.axes()
    assert np.allclose(forward, [0.0, 0.0, 1.0])


def test_spatial_graph_numeric_and_count_helpers():
    graph = SpatialGraph({
        "tv": {
            "position": [0.0, 0.0, -3.0],
            "box3d_center": [0.0, 0.0, -3.0],
            "box3d_size": [3.0, 4.0, 12.0],
            "box3d_min": [-1.5, -2.0, -9.0],
            "box3d_max": [1.5, 2.0, 3.0],
            "orientation": [0.0, 0.0, -1.0],
        },
        "handle_1": {
            "position": [1.0, 0.0, -3.0],
            "box3d_center": [1.0, 0.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [0.5, -0.5, -3.5],
            "box3d_max": [1.5, 0.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "handle_2": {
            "position": [2.0, 0.0, -3.0],
            "box3d_center": [2.0, 0.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [1.5, -0.5, -3.5],
            "box3d_max": [2.5, 0.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        },
    })

    assert graph.screen_diagonal("tv") == 5.0
    assert graph.volume("tv") == 144.0
    assert graph.nodes_with_prefix("handle_") == ["handle_1", "handle_2"]
    assert graph.count_prefix("handle_") == 2


def test_spatial_graph_overlap_clearance_and_collision_helpers():
    graph = SpatialGraph({
        "moving": {
            "position": [0.0, 1.0, 0.0],
            "box3d_center": [0.0, 1.0, 0.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, 0.5, -0.5],
            "box3d_max": [0.5, 1.5, 0.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "target": {
            "position": [0.0, 1.0, -3.0],
            "box3d_center": [0.0, 1.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, 0.5, -3.5],
            "box3d_max": [0.5, 1.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "overlap": {
            "position": [0.0, 1.0, 0.0],
            "box3d_center": [0.0, 1.0, 0.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, 0.5, -0.5],
            "box3d_max": [0.5, 1.5, 0.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "side": {
            "position": [3.0, 1.0, -3.0],
            "box3d_center": [3.0, 1.0, -3.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [2.5, 0.5, -3.5],
            "box3d_max": [3.5, 1.5, -2.5],
            "orientation": [0.0, 0.0, -1.0],
        },
        "upper": {
            "position": [0.0, 3.0, 0.0],
            "box3d_center": [0.0, 3.0, 0.0],
            "box3d_size": [1.0, 1.0, 1.0],
            "box3d_min": [-0.5, 2.5, -0.5],
            "box3d_max": [0.5, 3.5, 0.5],
            "orientation": [0.0, 0.0, -1.0],
        },
    })

    assert graph.footprint_overlap("moving", "overlap")
    assert not graph.footprint_overlap("moving", "side")
    assert graph.vertical_clearance("upper", "moving") == 1.0
    assert graph.would_collide_along_direction("moving", "target", [0.0, 0.0, -1.0])
    assert not graph.would_collide_along_direction("moving", "side", [0.0, 0.0, -1.0])
    assert not graph.would_collide_along_direction("moving", "target", [0.0, 0.0, 1.0])

