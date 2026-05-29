import os
from agent.prompt.template import task_description, api_specification, example_problems, code_generation_prompt, code_repair_prompt
from pySpatial_Interface import Scene


def _available_graph_objects_prompt(scene: Scene) -> str:
    graph = getattr(scene, "spatial_graph", None)
    if graph is None or not hasattr(graph, "list_nodes"):
        return ""
    names = graph.list_nodes()
    if not names:
        return ""
    object_lines = "\n".join(f"        - {name}" for name in names)
    return f"""
        Available graph object names:
{object_lines}

        Use these exact names in graph calls. Do not shorten names such as "leftmost cabinet" to "left", and do not rewrite names into snake_case.
        If the question wording differs from the available object names, choose the closest complete object name from this list.
    """


# TODO: Rewrite the codeAgent with structured output with pydantic


def _base_code_prompt() -> str:
    return f"""
        {task_description}
        {api_specification}
        {example_problems}
    """


def _call_code_model(query_for_vlm: str, api_key: str = None, backend: str = "local_qwen", model: str = "gpt-4.1", local_model_path: str = None, base_url: str = None, device: str = "cuda"):
    if backend == "local_qwen":
        from agent.model_backend import DEFAULT_LOCAL_QWEN_MODEL_PATH
        from agent.model_backend.local_qwen import generate_text_with_local_qwen

        return generate_text_with_local_qwen(
            query_for_vlm,
            model_path=local_model_path or DEFAULT_LOCAL_QWEN_MODEL_PATH,
            device=device,
            max_new_tokens=1000,
        )

    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Python code using the pySpatial API to solve spatial reasoning problems."},
                {"role": "user", "content": query_for_vlm}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {str(e)}")


def generate_code_from_query(scene: Scene, api_key: str = None, backend: str = "local_qwen", model: str = "gpt-4.1", local_model_path: str = None, base_url: str = None, device: str = "cuda"):
    """
    Generate code using the configured code model based on the scene question.
    """
    object_prompt = _available_graph_objects_prompt(scene)
    query_for_vlm = f"""
        {_base_code_prompt()}
        {object_prompt}
        {code_generation_prompt}
        the question is {scene.question}
    """

    return _call_code_model(
        query_for_vlm,
        api_key=api_key,
        backend=backend,
        model=model,
        local_model_path=local_model_path,
        base_url=base_url,
        device=device,
    )


def repair_code_from_error(
    scene: Scene,
    previous_response: str = None,
    previous_code: str = None,
    error: str = None,
    api_key: str = None,
    backend: str = "local_qwen",
    model: str = "gpt-4.1",
    local_model_path: str = None,
    base_url: str = None,
    device: str = "cuda",
):
    """Generate corrected code after parse or execution failure."""
    object_prompt = _available_graph_objects_prompt(scene)
    query_for_vlm = f"""
        {_base_code_prompt()}
        {object_prompt}
        {code_repair_prompt}

        Original question:
        {scene.question}

        Previous model response:
        {previous_response or "N/A"}

        Previously parsed code:
        ```python
        {previous_code or ""}
        ```

        Traceback or error:
        {error or "N/A"}
    """

    return _call_code_model(
        query_for_vlm,
        api_key=api_key,
        backend=backend,
        model=model,
        local_model_path=local_model_path,
        base_url=base_url,
        device=device,
    )

def generate_code(scene: Scene, api_key: str = None, **kwargs):
    """Legacy function name for backward compatibility"""
    return generate_code_from_query(scene, api_key, **kwargs)


