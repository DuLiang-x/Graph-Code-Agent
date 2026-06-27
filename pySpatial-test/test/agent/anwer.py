import os
import base64
from io import BytesIO
from PIL import Image
from agent.prompt.template import answer_background, answer_prompt, without_visual_clue_background
from pySpatial_Interface import Scene
import numpy as np
try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __repr__(self):
            fields = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
            return f"{self.__class__.__name__}({fields})"


def pil_image_to_data_url(pil: Image.Image) -> str:
    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_path_to_data_url(image_path: str) -> str:
    with Image.open(image_path) as image:
        return pil_image_to_data_url(image.convert("RGB"))


def _format_answer_object_context(scene: Scene, max_objects: int = 80) -> str:
    object_boxes = getattr(scene, "object_3d_boxes", None)
    if not isinstance(object_boxes, dict) or not object_boxes:
        return ""

    context_keys = (
        "box2d",
        "bbox_wh",
        "prompt_used",
        "counting_source_object",
        "counting_selection_source",
        "selection_decision",
        "selected_prompt",
    )
    lines = []
    for name, payload in object_boxes.items():
        if not isinstance(payload, dict):
            continue
        parts = []
        for key in context_keys:
            if key in payload and payload.get(key) is not None:
                parts.append(f"{key}={payload.get(key)}")
        if not parts:
            continue
        lines.append(f"- {name}: " + ", ".join(parts))
        if len(lines) >= max_objects:
            lines.append("... truncated ...")
            break
    return "\n".join(lines)


def _append_visual_clue_messages(messages, query_for_vlm: str, visual_clue):
    if visual_clue is None:
        messages.append({"role": "user", "content": query_for_vlm})
    elif isinstance(visual_clue, str):
        messages.append({"role": "user", "content": f"{query_for_vlm}\n\nVisual clue: {visual_clue}"})
    elif isinstance(visual_clue, dict):
        computed = visual_clue.get("computed_results") or visual_clue.get("results") or ""
        object_context = visual_clue.get("object_context") or ""
        object_context_text = f"\n\nObject boxes and node context for visual checks:\n{object_context}" if object_context else ""
        text_query = f"{query_for_vlm}\n\nComputed results: {computed}{object_context_text}"
        content = [{"type": "input_text", "text": text_query}]
        visualization_path = visual_clue.get("visualization_path") or visual_clue.get("image_path")
        if visualization_path and os.path.exists(visualization_path):
            content.append({"type": "input_image", "image_url": image_path_to_data_url(visualization_path)})
        messages.append({"role": "user", "content": content})
    elif isinstance(visual_clue, list):
        messages.append({"role": "user", "content": query_for_vlm})
        for o3d_img in visual_clue:
            data_url = o3d_image_to_data_url(o3d_img)
            messages.append({"role": "user", "content": [{"type": "input_image", "image_url": data_url}]})
    else:
        messages.append({"role": "user", "content": query_for_vlm})
        data_url = o3d_image_to_data_url(visual_clue)
        messages.append({"role": "user", "content": [{"type": "input_image", "image_url": data_url}]})


def o3d_image_to_data_url(o3d_img: "o3d.geometry.Image") -> str:
    """Convert Open3D Image -> PNG data URL suitable for OpenAI vision input."""
    arr = np.asarray(o3d_img)  # Open3D Image to NumPy, noted that the data type is int8
    
    # Handle different image formats
    if len(arr.shape) == 3:  # Color image
        if arr.shape[2] == 3:
            pil = Image.fromarray(arr, mode="RGB")
        elif arr.shape[2] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            raise ValueError(f"Unexpected number of channels {arr.shape[2]}")
    elif len(arr.shape) == 2:  # Grayscale image
        pil = Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Unexpected image shape {arr.shape}")
    
    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class SpatialAnswer(BaseModel):
    reasoning: str
    answer: str


def _coerce_spatial_answer(value):
    if isinstance(value, SpatialAnswer):
        return value
    if isinstance(value, dict):
        return SpatialAnswer(
            reasoning=str(value.get("reasoning", "")),
            answer=str(value.get("answer", "")).strip(),
        )
    if isinstance(value, str):
        from agent.model_backend.local_qwen import parse_spatial_answer_text

        return parse_spatial_answer_text(value)
    reasoning = getattr(value, "reasoning", None)
    answer_value = getattr(value, "answer", None)
    if reasoning is not None or answer_value is not None:
        return SpatialAnswer(reasoning=str(reasoning or ""), answer=str(answer_value or "").strip())
    return SpatialAnswer(reasoning=str(value), answer=str(value).strip())


def _extract_chat_completion_text(response) -> str:
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            first_choice = choices[0] or {}
            message = first_choice.get("message") or {}
            content = message.get("content") or first_choice.get("text")
            if content is not None:
                return content

    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is not None:
            return content
        text = getattr(first_choice, "text", None)
        if text is not None:
            return text

    raise TypeError("Unsupported chat completion response type: {}".format(type(response).__name__))


def _to_chat_completion_messages(messages):
    chat_messages = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chat_messages.append({"role": message.get("role", "user"), "content": content})
            continue

        converted = []
        text_parts = []
        has_image = False
        for item in content:
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text", "")
                text_parts.append(text)
                converted.append({"type": "text", "text": text})
            elif item_type in {"image", "input_image"}:
                image_url = item.get("image_url")
                image = item.get("image")
                if image_url is None and image is not None:
                    image_url = pil_image_to_data_url(image)
                if image_url:
                    has_image = True
                    converted.append({"type": "image_url", "image_url": {"url": image_url}})

        chat_content = converted if has_image else "\n".join(text_parts)
        chat_messages.append({"role": message.get("role", "user"), "content": chat_content})
    return chat_messages


def _call_openai_answer_model(client, model: str, messages, max_tokens: int = 2000):
    try:
        response = client.responses.parse(
            model=model,
            input=messages,
            max_output_tokens=max_tokens,
            text_format=SpatialAnswer,
        )
        return _coerce_spatial_answer(response.output_parsed)
    except Exception as responses_error:
        try:
            chat_messages = _to_chat_completion_messages(messages)
            chat_messages.append({
                "role": "user",
                "content": "Return JSON with keys reasoning and answer. The answer value must be the final open-form answer.",
            })
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            return _coerce_spatial_answer(_extract_chat_completion_text(response))
        except Exception as chat_error:
            raise Exception(
                "Error calling OpenAI-compatible answer API: responses.parse failed: {}; chat.completions failed: {}".format(
                    responses_error, chat_error
                )
            )


def answer(scene: Scene, api_key: str = None, backend: str = "local_qwen", model: str = "gpt-5", local_model_path: str = None, base_url: str = None, device: str = "cuda"):
    """
    Generate structured answer based on the scene question and visual clue.
    """
    base_prompt = f"""
        {answer_background}
        {answer_prompt}
    """

    object_context = _format_answer_object_context(scene)
    object_context_text = f"Object boxes and node context for visual checks:\n{object_context}" if object_context else "Object boxes and node context for visual checks: unavailable"

    query_for_vlm = f"""
        {base_prompt}
        the question is {scene.question}
        the generated code is {scene.code}
        the visual clue may include computed 3D results and optional visualization. Answer with the final open-form Omni3D-Bench answer, not an option letter.
        If computed_results contains needs_visual_color_check, needs_visual_visibility_check, needs_visual_clock_reading, or needs_visual_apparent_size_check, use the image, visualization, and object boxes below to complete the visual judgment. Do not output the needs_visual_* marker as the final answer.
        {object_context_text}
        The visual clue is pasted below:

    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes visual information and answers spatial reasoning questions based on the generated visual clue."}
    ]
    _append_visual_clue_messages(messages, query_for_vlm, scene.visual_clue)

    if backend == "local_qwen":
        from agent.model_backend import DEFAULT_LOCAL_QWEN_MODEL_PATH
        from agent.model_backend.local_qwen import answer_with_local_qwen, parse_spatial_answer_text

        raw = answer_with_local_qwen(
            messages,
            model_path=local_model_path or DEFAULT_LOCAL_QWEN_MODEL_PATH,
            device=device,
            max_new_tokens=2000,
        )
        parsed = parse_spatial_answer_text(raw)
        print(f"--------------------------------{parsed}")
        return parsed

    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    parsed = _call_openai_answer_model(client, model, messages, max_tokens=2000)

    print(f"--------------------------------{parsed}")
    return parsed

def answer_without_visual_clue(scene: Scene, api_key: str = None, backend: str = "local_qwen", model: str = "gpt-5", local_model_path: str = None, base_url: str = None, device: str = "cuda"):
    """Basic QA fallback: answer the question using only the images and question,
    without the pySpatial framework (no reconstruction, no code generation)."""
    base_prompt = f"""
        {without_visual_clue_background}
        the question is {scene.question}
        Answer with the final open-form Omni3D-Bench answer, not an option letter.
    """

    # Build message content with images
    content = [{"type": "input_text", "text": base_prompt}]

    for image_path in scene.images:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            ext = os.path.splitext(image_path)[1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{b64}",
            })

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers spatial reasoning questions based on the provided images."},
        {"role": "user", "content": content}
    ]

    if backend == "local_qwen":
        from agent.model_backend import DEFAULT_LOCAL_QWEN_MODEL_PATH
        from agent.model_backend.local_qwen import answer_with_local_qwen, parse_spatial_answer_text

        raw = answer_with_local_qwen(
            messages,
            model_path=local_model_path or DEFAULT_LOCAL_QWEN_MODEL_PATH,
            device=device,
            max_new_tokens=2000,
        )
        parsed = parse_spatial_answer_text(raw)
        print(f"[basic_qa] {parsed}")
        return parsed

    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    parsed = _call_openai_answer_model(client, model, messages, max_tokens=2000)

    print(f"[basic_qa] {parsed}")
    return parsed
