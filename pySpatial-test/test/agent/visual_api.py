import base64
import json
import os
import re
from io import BytesIO
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageDraw


DEFAULT_COLOR_CHOICES = [
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "beige",
    "tan",
    "gold",
    "silver",
]


class VisualAttributeClient:
    def __init__(
        self,
        backend: str = "local_qwen",
        api_key: str = None,
        model: str = "gpt-4.1",
        local_model_path: str = None,
        base_url: str = None,
        device: str = "cuda",
    ):
        self.backend = backend
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.local_model_path = local_model_path
        self.base_url = base_url
        self.device = device

    def classify_color(self, scene, object_name: str, choices: Optional[Iterable[str]] = None) -> str:
        result = self.classify_color_batch(scene, [object_name], choices=choices)
        return result.get(object_name, "unknown")

    def classify_color_batch(self, scene, object_names: Iterable[str], choices: Optional[Iterable[str]] = None) -> Dict[str, str]:
        names = [str(name) for name in object_names if str(name)]
        if not names:
            return {}

        prepared = _prepare_objects(scene, names)
        result = {name: "unknown" for name in names}
        if not prepared:
            return result

        choice_list = _normalize_choices(choices)
        overlay = _make_overlay(scene.images[0], prepared)
        crop_grid = _make_crop_grid(scene.images[0], prepared)
        prompt = _build_color_prompt(prepared, choice_list)
        response = self._process_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": overlay},
                        {"type": "image", "image": crop_grid},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_new_tokens=256,
        )
        parsed = _parse_color_response(response, [item["name"] for item in prepared], choice_list)
        result.update(parsed)
        return result

    def _process_messages(self, messages, max_new_tokens: int = 256) -> str:
        if self.backend == "local_qwen":
            from agent.model_backend import DEFAULT_LOCAL_QWEN_MODEL_PATH
            from agent.model_backend.local_qwen import get_local_qwen_client

            client = get_local_qwen_client(
                model_path=self.local_model_path or DEFAULT_LOCAL_QWEN_MODEL_PATH,
                device=self.device,
            )
            return client.process_messages(messages, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0).strip()

        if not self.api_key:
            return ""
        from openai import OpenAI

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        content = []
        for item in messages[0].get("content", []):
            if item.get("type") == "text":
                content.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                content.append({"type": "input_image", "image_url": _pil_to_data_url(item.get("image"))})
        response = client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_new_tokens,
        )
        return getattr(response, "output_text", str(response)).strip()


def _prepare_objects(scene, object_names: List[str]) -> List[Dict[str, object]]:
    if not getattr(scene, "images", None):
        return []
    image_path = scene.images[0]
    if not image_path or not os.path.exists(image_path):
        return []
    boxes = getattr(scene, "object_3d_boxes", None)
    if not isinstance(boxes, dict):
        return []
    prepared = []
    for name in object_names:
        payload = boxes.get(name)
        if not isinstance(payload, dict):
            continue
        box = payload.get("box2d")
        if not box or len(box) != 4:
            continue
        prepared.append({"name": name, "box2d": [int(v) for v in box]})
    return prepared


def _normalize_choices(choices: Optional[Iterable[str]]) -> List[str]:
    values = [str(choice).strip().lower() for choice in (choices or DEFAULT_COLOR_CHOICES)]
    values = [value for value in values if value]
    if "unknown" not in values:
        values.append("unknown")
    return values


def _build_color_prompt(objects: List[Dict[str, object]], choices: List[str]) -> str:
    names = [item["name"] for item in objects]
    return f"""
Identify the dominant visible color of each numbered target object.

Use the full image with boxes for context and the crop grid for object details.
Return only a JSON object whose keys are exactly these object names:
{json.dumps(names)}

Allowed color values:
{json.dumps(choices)}

Rules:
- Choose the closest allowed color for each object.
- Use "unknown" only if the target object is not visible enough to judge.
- Do not explain.
"""


def _make_overlay(image_path: str, objects: List[Dict[str, object]]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for idx, item in enumerate(objects, start=1):
        box = _clamp_box(item["box2d"], image.size)
        draw.rectangle(box, outline="red", width=5)
        draw.rectangle([box[0], max(0, box[1] - 26), box[0] + 42, box[1]], fill="white")
        draw.text((box[0] + 8, max(0, box[1] - 23)), str(idx), fill="red")
    return image


def _make_crop_grid(image_path: str, objects: List[Dict[str, object]]) -> Image.Image:
    source = Image.open(image_path).convert("RGB")
    tile_w, tile_h = 224, 250
    cols = min(4, max(1, len(objects)))
    rows = (len(objects) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(grid)
    for idx, item in enumerate(objects):
        row, col = divmod(idx, cols)
        x, y = col * tile_w, row * tile_h
        crop = source.crop(tuple(_clamp_box(item["box2d"], source.size))).resize((tile_w, 224))
        grid.paste(crop, (x, y + 26))
        label = f"{idx + 1}: {item['name']}"
        draw.rectangle([x, y, x + tile_w, y + 26], fill="white")
        draw.text((x + 5, y + 5), label[:36], fill="red")
    return grid


def _parse_color_response(response: str, names: List[str], choices: List[str]) -> Dict[str, str]:
    text = str(response or "").strip()
    data = None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1) if match else text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else text
    try:
        data = json.loads(candidate)
    except Exception:
        data = {}
    choices_set = set(choices)
    result = {}
    if isinstance(data, dict):
        for name in names:
            value = str(data.get(name, "unknown")).strip().lower()
            result[name] = value if value in choices_set else _closest_choice(value, choices_set)
    return result


def _closest_choice(value: str, choices: set) -> str:
    if value in {"grey"} and "gray" in choices:
        return "gray"
    for choice in choices:
        if choice != "unknown" and choice in value:
            return choice
    return "unknown"


def _clamp_box(box, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def _pil_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
