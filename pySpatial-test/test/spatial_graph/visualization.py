from __future__ import annotations

from object_3d_extraction.visualize_3d_aabb import render_3d_aabb_scene


def visualize_graph(graph, output_path: str, width: int = 1200, height: int = 700):
    return render_3d_aabb_scene(graph.to_extraction_results(), output_path, width=width, height=height)
