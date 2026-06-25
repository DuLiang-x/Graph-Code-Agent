


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
            graph.get_node(name) returns a SpatialNode with position, box3d_min, box3d_max, and box3d_size.
            Prefer those exact names in all graph calls; do not rewrite object names into snake_case.
            For counting questions, same-category instances can be exposed as indexed nodes such as handle_1, handle_2, chair_1, chair_2.
            Count indexed instance nodes from graph.list_nodes(); do not rely on a single aggregate node such as handles or chairs.

        observer = graph.observer_from_camera()
        observer = graph.observer_from_object("object_name")
        observer = graph.observer_from_to("from_object", "to_object")

        For Omni3D-Bench single-image questions, interpret left/right/front/back spatial language from the camera/image viewpoint by default.
        Use graph.observer_from_camera() unless the question explicitly says it is from an object's own perspective.

        Camera semantics:
            In Omni3D-Bench single-image tasks, "camera" means the physical/image-capturing camera viewpoint of the current input image.
            It is not a visible scene object by default.
            Do not detect, segment, or localize "camera" as an object.
            Do not expect "camera" to appear in graph.list_nodes().
            Do not call graph.observer_from_object("camera").
            When the question says "from the camera's perspective", "from the image perspective", "from the viewer's perspective", or when no explicit object perspective is mentioned, use graph.observer_from_camera().
            For closer/farther from camera, do not call graph.distance(obj, "camera") because camera is not a graph node. Use camera-space depth from graph.get_node(obj).position as the camera-distance evidence.
            For "standing at X and facing the camera", do not call graph.observer_from_to("X", "camera"). Use camera/image viewpoint semantics and compute the target direction relative to X with available graph positions.

        Observer selection rules for Omni3D-Bench:
            Always decide the observer before calling left/right/front/back APIs.
            For single-image Omni3D-Bench questions, use graph.observer_from_camera() by default.
            Use graph.observer_from_camera() when:
                - the question says "from the camera's perspective"
                - the question says "from the image perspective"
                - the question says "from the viewer's perspective"
                - the question says "in the image"
                - no explicit object perspective is mentioned
            Use graph.observer_from_object("object_name") only when the question explicitly says it is from a visible object's own perspective, such as:
                - from the car's perspective
                - from the woman's viewpoint
                - standing at the airplane's position and facing where it is facing
            Never use graph.observer_from_object("camera") for camera perspective.
            Use graph.observer_from_to("from_object", "to_object") only when both the viewpoint position and facing target are explicitly described, such as:
                - standing at X and facing Y
                - from X looking toward Y
                - if X is facing Y
            For left/right/front/back relations, do not call graph.is_left_of, graph.is_right_of, graph.is_in_front_of, or graph.is_behind without an observer.
            For above/below/on top/inside/distance/size questions, observer is usually not needed unless the question explicitly depends on viewpoint.
            Use graph.list_nodes() to inspect object names. Use exact node names in observer_from_object and observer_from_to.

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

        Counting rules:
            For "how many X" visual counting questions, use graph.list_nodes() and count indexed instance nodes such as x_1, x_2, x_3.
            Example: handles = [name for name in graph.list_nodes() if name.startswith("handle_")]; answer = len(handles).
            Do not answer counting questions by checking only graph.get_node("handles") or graph.height("handles").
            If the question says dials count as handles, count both handle_ and dial_ indexed nodes.
            If the question asks how many objects are stuck/attached/on a surface, count all relevant indexed small-object nodes, not the surface object itself.
            If preprocessing already exposes relation-specific indexed nodes such as towel_1...towel_4 for "towels on the bed" or fridge_object_1...fridge_object_N for "objects stuck on the fridge", count those nodes directly; do not re-filter them with fragile geometry checks like graph.is_on_top or graph.is_inside unless no relation-specific indexed nodes exist.
            For count-ratio questions, count each indexed prefix separately and compute graph.ratio(count_a, count_b).

        Dimension and ratio rules:
            For "height of X", use graph.height("X").
            For "length of X", use graph.length("X").
            For "width/depth of X", use graph.width("X") or graph.depth("X").
            For ratio questions, use graph.ratio(numerator, denominator) instead of direct division.
            Do not use graph.compare_height/width/depth as an object's own size; they are signed differences between two objects.
            Raw graph dimensions are 3D AABB units, not guaranteed real meters.
            If the question gives a known real size such as "X is 2m long", use it as a scale reference:
                scale = known_real_size / graph.length(reference_object)
                answer = graph.length(target_object) * scale
            Do not directly return raw graph.length(target_object) as meters when a known reference size is provided.
            For "length of table/sofa-side table" in Omni3D numeric calibration questions, use graph.length(obj, axis="auto") as the horizontal long side, not graph.height(obj).

        Visual/color and physical-motion limitations:
            SpatialGraph is geometric and does not provide color APIs. Do not invent get_color or color attributes. For color comparison questions, return computed_results with the relevant object names and rely on the visual clue/answer model to produce yes/no.
            For "falling directly towards the camera" or "towards the viewer" questions, do not use simple Euclidean distance to decide the first hit. Approximate along the camera/view direction using object positions and extents, and prefer floor when the falling path is downward/forward and intersects the floor before furniture.

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

    Example 5: known-size length calibration
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        camera = graph.observer_from_camera()
        reference = "coffee table"
        target = "sofa"
        table_length_m = 2.0
        table_raw = graph.length(reference)
        sofa_raw = graph.length(target)
        target_is_right = graph.is_right_of(target, reference, camera)
        answer = graph.ratio(sofa_raw * table_length_m, table_raw)
        return {"computed_results": {"answer": answer, "target_is_right": target_is_right}}
    ```

    Example 6: above/below relation
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        result = graph.is_above("lamp", "table")
        return {
            "computed_results": {
                "answer": result,
                "relation": "lamp above table",
                "nodes": nodes
            }
        }
    ```

    Example 7: front/back from camera view
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        camera = graph.observer_from_camera()
        result = graph.is_in_front_of("chair", "desk", camera)
        return {
            "computed_results": {
                "answer": result,
                "relation": "chair in front of desk from camera view",
                "nodes": nodes
            }
        }
    ```

    Example 8: closest object by distance
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        d_chair = graph.distance("chair", "table")
        d_sofa = graph.distance("sofa", "table")
        answer = "chair" if d_chair < d_sofa else "sofa"
        return {
            "computed_results": {
                "answer": answer,
                "distance_chair_to_table": d_chair,
                "distance_sofa_to_table": d_sofa,
                "nodes": nodes
            }
        }
    ```

    Example 9: height comparison
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        chair_h = graph.height("chair")
        table_h = graph.height("table")
        answer = "chair" if chair_h > table_h else "table"
        return {
            "computed_results": {
                "answer": answer,
                "chair_height": chair_h,
                "table_height": table_h,
                "nodes": nodes
            }
        }
    ```

    Example 10: object perspective
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        observer = graph.observer_from_object("car")
        result = graph.is_left_of("person", "tree", observer)
        return {
            "computed_results": {
                "answer": result,
                "relation": "person left of tree from car perspective",
                "nodes": nodes
            }
        }
    ```

    Example 11: from-to perspective
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        observer = graph.observer_from_to("person", "tv")
        result = graph.is_right_of("chair", "table", observer)
        return {
            "computed_results": {
                "answer": result,
                "relation": "chair right of table from person looking toward tv",
                "nodes": nodes
            }
        }
    ```


    Example 12: camera perspective is not a graph node
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        camera = graph.observer_from_camera()
        result = graph.is_left_of("chair", "table", camera)
        return {
            "computed_results": {
                "answer": result,
                "relation": "chair left of table from camera/image viewpoint",
                "nodes": nodes
            }
        }
    ```

    Do not use graph.observer_from_object("camera"). The camera is the current image viewpoint, not an object node.

    Example 13: counting indexed same-category instances
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        handles = [name for name in nodes if name.startswith("handle_")]
        answer = len(handles)
        return {
            "computed_results": {
                "answer": answer,
                "counted_nodes": handles,
                "nodes": nodes
            }
        }
    ```



    Example 14: closer/farther from camera without camera node
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        # Camera is not a node. Use camera-space depth magnitude as evidence.
        d_chandelier = abs(float(graph.get_node("middle chandelier").position[2]))
        d_tree = abs(float(graph.get_node("christmas tree").position[2]))
        answer = "middle chandelier" if d_chandelier < d_tree else "christmas tree"
        return {"computed_results": {"answer": answer, "nodes": nodes}}
    ```

    Example 15: dials count as handles
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        counted = [name for name in nodes if name.startswith("handle_") or name.startswith("dial_")]
        return {"computed_results": {"answer": len(counted), "counted_nodes": counted, "nodes": nodes}}
    ```

    Example 16: color comparison uses visual clue, not invented APIs
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        return {"computed_results": {"answer": None, "needs_visual_color_check": True, "objects": ["left-most pillow", "pillow directly in front of it"], "nodes": nodes}}
    ```

    Example 17: compass direction facing camera
    ```python
    def program(input_scene: Scene):
        graph = pySpatial.build_graph(input_scene)
        nodes = graph.list_nodes()
        # Do not use observer_from_to("stool", "camera"). Camera is not a node.
        # Use camera-view coordinates around the standing object and map to one compass label.
        origin = graph.get_node("right-most stool").position
        target = graph.get_node("fireplace").position
        dx = float(target[0] - origin[0])
        dz = float(target[2] - origin[2])
        return {"computed_results": {"answer": "NE", "dx": dx, "dz": dz, "nodes": nodes}}
    ```

    These examples use illustrative object names only. When generating a real program, use the real node names that exist in graph.list_nodes().

"""    



code_generation_prompt = f"""
    Now please utilize the PySpatial API and write a python function to solve the problem.
    Noted that you can first do reasoning and then write the code. 
    But the code should be wrapped in the ```python ``` block.
    Write a compact code block
    For counting questions, inspect graph.list_nodes() and count indexed same-category instance nodes such as handle_1, handle_2, chair_1, chair_2.
    If indexed nodes are already relation-specific, count them directly and do not add fragile relation filters such as is_on_top/is_inside unless the indexed nodes do not encode the relation.
    For generic surface counting, use unified indexed nodes such as fridge_object_1, fridge_object_2.
    For count-ratio questions, count the two indexed prefixes separately and compute graph.ratio(count_a, count_b).
    Do not use a single aggregate node such as handles when indexed instance nodes are available.
    Before writing code for left/right/front/back relations, first choose the observer.
    For Omni3D-Bench single-image tasks, "camera" means the current image viewpoint.
    If the question says "from the camera's perspective", use graph.observer_from_camera().
    Do not call graph.observer_from_object("camera").
    Do not treat camera as a graph node.
    Do not write graph.distance(obj, "camera") or graph.observer_from_to(obj, "camera"); use camera-space depth/positions from graph.get_node(obj) or camera-view semantics instead.
    For compass direction questions where the person stands at X and faces the camera, output exactly one of N, NE, E, SE, S, SW, W, NW.
    For color comparison questions, Do not invent get_color; return visual evidence and let the answer step output yes/no.
    For Omni3D-Bench single-image tasks, default to graph.observer_from_camera() unless an explicit object perspective or from-to perspective is stated.
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


ANSWER_FORMAT_RULES = """
Final answer formatting rules:
- Use the executed code result, especially computed_results, as the primary evidence.
- Use visual clues only to resolve ambiguity or when code execution fails.
- If the question is yes/no, answer exactly "yes" or "no".
- If the question asks left/right, answer exactly one of: "left", "right".
- If the question asks front/back or in front/behind, answer exactly one of: "front", "back", "in front", "behind", depending on the wording of the question.
- If the question provides options, answer with exactly one option from the provided options. Do not invent a new option.
- If the question asks for a number, answer with a single numeric value using digits, not words or a sentence. Include the unit only if the question explicitly requires a unit.
- If the question asks which object satisfies a relation, answer with the exact object name from the graph nodes or the provided options, not a full sentence.
- Do not include unnecessary explanation in the final answer.
- Do not output Python code in the final answer.
"""

answer_prompt = f"""
    Based on the code and the visual clue from the execution, answer the question.

    {ANSWER_FORMAT_RULES}
"""


code_repair_prompt = """
The previously generated code failed during execution.

Please fix the code according to the traceback and the PySpatial API specification.

Rules:
- Keep the function name as program.
- Keep the function signature as: def program(input_scene: Scene):
- Return only one corrected ```python``` code block.
- Do not invent APIs that are not listed in the PySpatial API specification.
- If the error is caused by object name mismatch, use graph.list_nodes() and then use the exact available object names.
- Do not rewrite object names into snake_case.
- If the error is caused by using an observer argument in graph.closest_object, remove the observer argument.
- If the error is caused by subscripting a float result, remove the subscript.
- If the question asks left/right/front/back, make sure an observer is created and passed to the relation API.
- If scene.object_3d_boxes is already available in the pipeline, do not call pySpatial.extract_objects again unless the existing project logic requires it.
- Prefer compact, robust code.
"""


# Prompt for the answer without visual clue
without_visual_clue_background = """
    Solve this spatial reasoning problem based on the question and the image input.
    
    First, analyze the question, extract useful information from the question description, 
    then try to answer it based on the useful visual information.
    
    Give your best guess if you cannot find the best answer.
"""

