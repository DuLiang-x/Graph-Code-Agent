from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Watch:
    name: str
    condition_fn: Callable[[Any], Any]
    last_value: Any = None

    def evaluate(self, graph):
        self.last_value = self.condition_fn(graph)
        return self.last_value


class HookSystem:
    def __init__(self, graph):
        self.graph = graph
        self.watches: Dict[str, Watch] = {}
        self.events: List[dict] = []
        self.node_move_callbacks: List[Callable[..., Any]] = []
        self.property_callbacks: Dict[str, List[Callable[..., Any]]] = {}

    def add_watch(self, watch: Watch):
        self.watches[watch.name] = watch
        return watch

    def on_node_move(self, callback):
        self.node_move_callbacks.append(callback)
        return callback

    def on_property_change(self, property_name: str, callback):
        self.property_callbacks.setdefault(property_name, []).append(callback)
        return callback

    def check_condition(self, condition_fn):
        return condition_fn(self.graph)

    def record_event(self, event_name: str, data=None):
        event = {"event": event_name, "data": data or {}}
        self.events.append(event)
        return event

    def notify_node_move(self, name: str, old_position, new_position):
        for callback in self.node_move_callbacks:
            callback(name, old_position, new_position)

    def evaluate_watches(self):
        return {name: watch.evaluate(self.graph) for name, watch in self.watches.items()}
