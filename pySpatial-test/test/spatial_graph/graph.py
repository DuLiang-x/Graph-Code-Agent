from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=float)


def _as_vector(value, default=None) -> np.ndarray:
    if value is None:
        value = default
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError("Expected a 3D vector")
    return arr[:3].astype(float)


def _normalize(value, fallback=None) -> np.ndarray:
    vec = _as_vector(value, fallback)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        if fallback is None:
            raise ValueError("Cannot normalize zero vector")
        vec = _as_vector(fallback)
        norm = float(np.linalg.norm(vec))
    return vec / norm


@dataclass
class Observer:
    position: np.ndarray
    look_at: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position = _as_vector(self.position)
        if self.look_at is not None:
            self.look_at = _as_vector(self.look_at)
        if self.orientation is not None:
            self.orientation = _normalize(self.orientation, fallback=[0.0, 0.0, -1.0])
        if self.look_at is None and self.orientation is None:
            self.orientation = np.array([0.0, 0.0, -1.0], dtype=float)

    @classmethod
    def from_camera(cls):
        return cls(position=np.zeros(3), orientation=np.array([0.0, 0.0, -1.0]))

    @classmethod
    def from_object(cls, node: "SpatialNode"):
        return cls(position=node.position, orientation=node.orientation)

    @classmethod
    def from_object_toward(cls, node: "SpatialNode", target: "SpatialNode"):
        return cls(position=node.position, look_at=target.position)

    def axes(self):
        if self.look_at is not None:
            fallback = self.orientation if self.orientation is not None else [0.0, 0.0, -1.0]
            forward = _normalize(self.look_at - self.position, fallback=fallback)
        else:
            forward = _normalize(self.orientation, fallback=[0.0, 0.0, -1.0])
        right = np.cross(forward, WORLD_UP)
        if np.linalg.norm(right) < 1e-9:
            right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        right = _normalize(right, fallback=[1.0, 0.0, 0.0])
        up = _normalize(np.cross(forward, right), fallback=WORLD_UP)
        return right, up, forward


@dataclass
class SpatialNode:
    name: str
    position: np.ndarray
    box3d_center: np.ndarray
    box3d_size: np.ndarray
    box3d_min: np.ndarray
    box3d_max: np.ndarray
    orientation: np.ndarray

    @classmethod
    def from_extraction(cls, name: str, item: dict) -> "SpatialNode":
        position = _as_vector(item.get("position"), item.get("box3d_center"))
        center = _as_vector(item.get("box3d_center"), position)
        size = _as_vector(item.get("box3d_size"), [0.0, 0.0, 0.0])
        box_min = _as_vector(item.get("box3d_min"), center - size / 2.0)
        box_max = _as_vector(item.get("box3d_max"), center + size / 2.0)
        orientation = _normalize(item.get("orientation"), fallback=[0.0, 0.0, -1.0])
        return cls(str(name), position, center, size, box_min, box_max, orientation)

    def to_extraction(self) -> dict:
        return {
            "position": self.position.tolist(),
            "box3d_center": self.box3d_center.tolist(),
            "box3d_size": self.box3d_size.tolist(),
            "box3d_min": self.box3d_min.tolist(),
            "box3d_max": self.box3d_max.tolist(),
            "orientation": self.orientation.tolist(),
        }


NodeRef = Union[str, SpatialNode]


def _normalize_node_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").split()).lower()


class SpatialGraph:
    def __init__(self, nodes: Optional[Union[Dict[str, dict], Iterable[SpatialNode]]] = None):
        self.nodes: Dict[str, SpatialNode] = {}
        if isinstance(nodes, dict):
            for name, item in nodes.items():
                if isinstance(item, SpatialNode):
                    self.add_node(item)
                elif isinstance(item, dict) and not item.get("error"):
                    self.add_node(SpatialNode.from_extraction(name, item))
        elif nodes is not None:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node: SpatialNode):
        self.nodes[node.name] = node
        return node

    def remove_node(self, name: str):
        return self.nodes.pop(name)

    def get_node(self, node: NodeRef) -> SpatialNode:
        if isinstance(node, SpatialNode):
            return node
        name = str(node)
        if name in self.nodes:
            return self.nodes[name]

        normalized_name = _normalize_node_name(name)
        for existing_name, existing_node in self.nodes.items():
            if _normalize_node_name(existing_name) == normalized_name:
                return existing_node

        available = ", ".join(self.list_nodes()) or "<none>"
        raise KeyError(f"Unknown object {name!r}. Available objects: {available}")

    def list_nodes(self) -> List[str]:
        return list(self.nodes.keys())

    def observer_from_camera(self):
        return Observer.from_camera()

    def observer_from_object(self, name: str):
        return Observer.from_object(self.get_node(name))

    def observer_from_to(self, from_obj: str, to_obj: str):
        return Observer.from_object_toward(self.get_node(from_obj), self.get_node(to_obj))

    def distance(self, a: NodeRef, b: NodeRef) -> float:
        return float(np.linalg.norm(self.get_node(a).position - self.get_node(b).position))

    def is_above(self, a: NodeRef, b: NodeRef, tolerance: float = 0.0) -> bool:
        return bool(self.get_node(a).position[1] > self.get_node(b).position[1] + tolerance)

    def is_below(self, a: NodeRef, b: NodeRef, tolerance: float = 0.0) -> bool:
        return bool(self.get_node(a).position[1] < self.get_node(b).position[1] - tolerance)

    def is_inside(self, inner: NodeRef, outer: NodeRef, tolerance: float = 1e-6) -> bool:
        a = self.get_node(inner)
        b = self.get_node(outer)
        return bool(np.all(a.box3d_min >= b.box3d_min - tolerance) and np.all(a.box3d_max <= b.box3d_max + tolerance))

    def is_on_top(self, a: NodeRef, b: NodeRef, tolerance: float = 0.05) -> bool:
        node_a = self.get_node(a)
        node_b = self.get_node(b)
        bottom_a = min(node_a.box3d_min[1], node_a.box3d_max[1])
        top_b = max(node_b.box3d_min[1], node_b.box3d_max[1])
        xy_overlap = (
            node_a.box3d_max[0] >= node_b.box3d_min[0]
            and node_a.box3d_min[0] <= node_b.box3d_max[0]
            and node_a.box3d_max[2] >= node_b.box3d_min[2]
            and node_a.box3d_min[2] <= node_b.box3d_max[2]
        )
        return bool(xy_overlap and abs(bottom_a - top_b) <= tolerance)

    def size_ratio(self, a: NodeRef, b: NodeRef) -> float:
        vol_a = float(np.prod(np.maximum(self.get_node(a).box3d_size, 0.0)))
        vol_b = float(np.prod(np.maximum(self.get_node(b).box3d_size, 0.0)))
        return vol_a / vol_b if vol_b else float("inf")

    def height(self, obj: NodeRef) -> float:
        return float(self.get_node(obj).box3d_size[1])

    def width(self, obj: NodeRef) -> float:
        return float(self.get_node(obj).box3d_size[0])

    def depth(self, obj: NodeRef) -> float:
        return float(self.get_node(obj).box3d_size[2])

    def length(self, obj: NodeRef, axis: str = "auto") -> float:
        node = self.get_node(obj)
        if axis in ("x", "width", "image_x"):
            return float(node.box3d_size[0])
        if axis in ("z", "depth", "depth_z"):
            return float(node.box3d_size[2])
        if axis != "auto":
            raise ValueError("axis must be 'auto', 'x'/'width'/'image_x', or 'z'/'depth'/'depth_z'")
        return float(max(node.box3d_size[0], node.box3d_size[2]))

    def ratio(self, numerator: float, denominator: float, eps: float = 1e-9) -> float:
        denominator = float(denominator)
        if abs(denominator) < eps:
            return float("inf")
        return float(numerator) / denominator

    def compare_height(self, a: NodeRef, b: NodeRef) -> float:
        return float(self.get_node(a).box3d_size[1] - self.get_node(b).box3d_size[1])

    def compare_width(self, a: NodeRef, b: NodeRef) -> float:
        return float(self.get_node(a).box3d_size[0] - self.get_node(b).box3d_size[0])

    def compare_depth(self, a: NodeRef, b: NodeRef) -> float:
        return float(self.get_node(a).box3d_size[2] - self.get_node(b).box3d_size[2])

    def closest_object(self, target: NodeRef, candidates: Optional[Sequence[NodeRef]] = None):
        names = candidates if candidates is not None else [name for name in self.nodes if name != self.get_node(target).name]
        if not names:
            return None
        return min(names, key=lambda name: self.distance(target, name))

    def relative_position(self, a: NodeRef, b: NodeRef, observer: Observer) -> str:
        node_a = self.get_node(a)
        node_b = self.get_node(b)
        right, up, forward = observer.axes()
        delta = node_a.position - node_b.position
        scores = {
            "right": float(np.dot(delta, right)),
            "left": float(-np.dot(delta, right)),
            "above": float(np.dot(delta, up)),
            "below": float(-np.dot(delta, up)),
            "front": float(-np.dot(node_a.position - observer.position, forward) + np.dot(node_b.position - observer.position, forward)),
            "back": float(np.dot(node_a.position - observer.position, forward) - np.dot(node_b.position - observer.position, forward)),
        }
        return max(scores, key=scores.get)

    def is_left_of(self, a: NodeRef, b: NodeRef, observer: Observer) -> bool:
        right, _, _ = observer.axes()
        return bool(np.dot(self.get_node(a).position - self.get_node(b).position, right) < 0)

    def is_right_of(self, a: NodeRef, b: NodeRef, observer: Observer) -> bool:
        right, _, _ = observer.axes()
        return bool(np.dot(self.get_node(a).position - self.get_node(b).position, right) > 0)

    def is_in_front_of(self, a: NodeRef, b: NodeRef, observer: Observer) -> bool:
        _, _, forward = observer.axes()
        da = np.dot(self.get_node(a).position - observer.position, forward)
        db = np.dot(self.get_node(b).position - observer.position, forward)
        return bool(da < db)

    def is_behind(self, a: NodeRef, b: NodeRef, observer: Observer) -> bool:
        return not self.is_in_front_of(a, b, observer)

    def angular_offset(self, a: NodeRef, b: NodeRef, observer: Observer) -> float:
        right, _, forward = observer.axes()
        delta = self.get_node(a).position - self.get_node(b).position
        return float(np.degrees(np.arctan2(np.dot(delta, right), np.dot(delta, forward))))

    def objects_in_view(self, observer: Observer, max_distance: Optional[float] = None) -> List[str]:
        _, _, forward = observer.axes()
        visible = []
        for name, node in self.nodes.items():
            delta = node.position - observer.position
            forward_dist = float(np.dot(delta, forward))
            dist = float(np.linalg.norm(delta))
            if forward_dist > 0 and (max_distance is None or dist <= max_distance):
                visible.append((dist, name))
        visible.sort()
        return [name for _, name in visible]

    def move_object(self, name: str, delta):
        node = self.get_node(name)
        delta_vec = _as_vector(delta)
        node.position = node.position + delta_vec
        node.box3d_center = node.box3d_center + delta_vec
        node.box3d_min = node.box3d_min + delta_vec
        node.box3d_max = node.box3d_max + delta_vec
        return node

    def copy_object(self, name: str, new_name: str):
        node = deepcopy(self.get_node(name))
        node.name = new_name
        return self.add_node(node)

    def scale_object(self, name: str, factor: float):
        node = self.get_node(name)
        node.box3d_size = node.box3d_size * float(factor)
        half = node.box3d_size / 2.0
        node.box3d_min = node.box3d_center - half
        node.box3d_max = node.box3d_center + half
        return node

    def rotate_object(self, name: str, angle: float, axis: str = "y"):
        node = self.get_node(name)
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        if axis == "x":
            rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
        elif axis == "z":
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        else:
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
        node.orientation = _normalize(rot.dot(node.orientation), fallback=[0.0, 0.0, -1.0])
        return node

    def snapshot(self):
        return deepcopy(self.nodes)

    def restore(self, snapshot):
        self.nodes = deepcopy(snapshot)

    @contextmanager
    def with_state(self):
        snap = self.snapshot()
        try:
            yield self
        finally:
            self.restore(snap)

    def to_extraction_results(self) -> dict:
        return {name: node.to_extraction() for name, node in self.nodes.items()}
