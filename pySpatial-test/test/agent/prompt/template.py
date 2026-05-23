


task_description = """
    You are now asked to solve a spatial reasoning related problem. The input are image(s) and a natural langugae question that
    specifically designed to test your spatial reasoning ability.
    It is not trivial to solve these tasks directly as a vision langugae model. 
    However, You have access to the following Python API:
"""

api_specification = """
    PySpatial exposes two compatible reasoning paths.

    Graph path (preferred for spatial reasoning):
        pySpatial.extract_objects(scene, device="cuda", mask_fallback="auto")
            Extracts question-relevant objects into scene.object_3d_boxes.
            The implementation uses the local object_3d_extraction package at
            /data/duliang/pySpatial-test/test/object_3d_extraction.
            mask_fallback="auto" reads/writes /data/duliang/pySpatial-test/outputs/Omni3D-Bench.
            mask_fallback="off" reads/writes /data/duliang/pySpatial-test/outputs/Omni3D-Benchnomask.

        graph = pySpatial.build_graph(scene)
            Builds a SpatialGraph from scene.object_3d_boxes.
            Use graph.list_nodes() to inspect the exact object names available in the graph.
            Prefer those exact names in all graph calls; do not rewrite object names into snake_case.

        observer = graph.observer_from_camera()
        observer = graph.observer_from_object("object_name")
        observer = graph.observer_from_to("from_object", "to_object")

        For Omni3D-Bench single-image questions, interpret left/right/front/back spatial language from the camera/image viewpoint by default.
        Use graph.observer_from_camera() unless the question explicitly says it is from an object's own perspective.

        SpatialGraph query APIs:
            graph.distance(a, b) -> float
            graph.is_above(a, b), graph.is_below(a, b)
            graph.is_inside(inner, outer), graph.is_on_top(a, b)
            graph.size_ratio(a, b) -> float
            graph.height(obj) -> float
            graph.width(obj) -> float
            graph.depth(obj) -> float
            graph.length(obj, axis="auto") -> float
            graph.ratio(numerator, denominator, eps=1e-9) -> float
            graph.compare_height(a, b) -> float
            graph.compare_width(a, b) -> float
            graph.compare_depth(a, b) -> float
                These compare_* APIs return signed size differences: size(a) - size(b).
                They may be negative and must not be used as an object's own height, width, depth, or length.
            graph.closest_object(target, candidates=None)
                Returns the nearest object name. It does not accept an observer argument.
                Do not write graph.closest_object(..., observer=camera).
            graph.relative_position(a, b, observer)
            graph.is_left_of(a, b, observer), graph.is_right_of(a, b, observer)
            graph.is_in_front_of(a, b, observer), graph.is_behind(a, b, observer)
            graph.angular_offset(a, b, observer) -> float
            graph.objects_in_view(observer, max_distance=None)

        Float-returning APIs produce numbers, not arrays or dictionaries. Do not subscript them:
            Correct: ratio = graph.size_ratio("tv", "table")
            Wrong: graph.size_ratio("tv", "table")[0]

        Dimension and ratio rules:
            For "height of X", use graph.height("X").
            For "length of X", use graph.length("X").
            For "width/depth of X", use graph.width("X") or graph.depth("X").
            For ratio questions, use graph.ratio(numerator, denominator) instead of direct division.
            Do not use graph.compare_height/width/depth as an object's own size; they are signed differences between two objects.

        Hypothetical reasoning APIs:
            graph.move_object(name, delta)
            graph.copy_object(name, new_name)
            graph.scale_object(name, factor)
            graph.rotate_object(name, angle, axis="y")
            graph.snapshot(), graph.restore(snapshot), graph.with_state()

        Visualization:
            pySpatial.visualize_graph(graph, output_path) returns a PNG path.

    Legacy reconstruction path remains available for comparison:
        reconstruction = pySpatial.reconstruct(scene)
        camera_motion = pySpatial.describe_camera_motion(reconstruction)
        pySpatial.synthesize_novel_view(reconstruction, new_camera_pose)

    Return a compact visual clue. For graph reasoning, prefer returning a dict:
        {
            "computed_results": {"answer": ..., "evidence": ...},
            "visualization_path": optional_png_path
        }

    please follow the instructions to generate the code in the ```python ``` block.
"""

# in-context learning exmaples
example_problems = """
    Example 1: left/right from camera view
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        camera = graph.observer_from_camera()
        result = graph.is_left_of("chair", "table", camera)
        return {"computed_results": {"chair_left_of_table": result}}
    ```

    Example 2: hypothetical move
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        camera = graph.observer_from_camera()
        with graph.with_state():
            graph.move_object("sofa", [2.0, 0.0, 0.0])
            result = graph.is_right_of("sofa", "tv", camera)
        return {"computed_results": {"after_move_sofa_right_of_tv": result}}
    ```

    Example 3: closest object
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nearest = graph.closest_object("table")
        return {"computed_results": {"closest_to_table": nearest}}
    ```

    Example 4: height ratio
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        fireplace_h = graph.height("fireplace")
        table_h = graph.height("coffee table")
        sofa_h = graph.height("sofa")
        ratio = graph.ratio(fireplace_h, table_h + sofa_h)
        return {"computed_results": {"answer": ratio}}
    ```

    Example 5: object to the right in an image
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        camera = graph.observer_from_camera()
        result = graph.is_right_of("sofa", "coffee table", camera)
        return {"computed_results": {"sofa_right_of_coffee_table": result}}
    ```
"""    



code_generation_prompt = f"""
    Now please utilize the PySpatial API and write a python function to solve the problem.
    Noted that you can first do reasoning and then write the code. 
    But the code should be wrapped in the ```python ``` block.
    Write a compact code block
    Also, the function written should be named as program and the input parameter should be a Scene object.
    for example,
    ```python
    def program(input_scene: Scene):
        ...
        return ...
    ```
"""


# Prompt template for ReAct: ReAct: Synergizing Reasoning and Acting in Language Models https://arxiv.org/abs/2210.03629

answer_background = f"""
    We are now solving a spatial reasoing problem.     
    It is not trivial to solve these tasks directly as a vision langugae model. 
    However, We have access to the following PySpatial API:
    {api_specification}
    
    We generate a python code based on the PySpatial API to solve this problem.
"""

answer_prompt = """
    Based on the code and the visual clue from the execution, answer the question.
"""




# Prompt for the answer without visual clue
without_visual_clue_background = """
    Solve this spatial reasoning problem based on the question and the image input.
    
    First, analyze the question, extract useful information from the question description, 
    then try to answer it based on the useful visual information.
    
    Give your best guess if you cannot find the best answer.
"""

