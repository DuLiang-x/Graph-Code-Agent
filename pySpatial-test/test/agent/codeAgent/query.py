import os
from agent.prompt.template import task_description, api_specification, example_problems, code_generation_prompt
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


def generate_code_from_query(scene: Scene, api_key: str = None, backend: str = "local_qwen", model: str = "gpt-4.1", local_model_path: str = None, base_url: str = None, device: str = "cuda"):
    """
    Generate code using OpenAI GPT-4 model based on the scene question.
    
    Args:
        scene: Scene object containing the question
        api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
    
    Returns:
        str: Generated code response from GPT-4
    """
    base_prompt = f"""
        {task_description}
        {api_specification}
        {example_problems}
    """

    object_prompt = _available_graph_objects_prompt(scene)
    query_for_vlm = f"""
        {base_prompt}
        {object_prompt}
        {code_generation_prompt}
        the question is {scene.question}
    """

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

def generate_code(scene: Scene, api_key: str = None, **kwargs):
    """Legacy function name for backward compatibility"""
    return generate_code_from_query(scene, api_key, **kwargs)


