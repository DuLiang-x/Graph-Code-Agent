from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import List, Optional

from PIL import Image

from . import DEFAULT_LOCAL_QWEN_MODEL_PATH


class LocalQwenClient:
    def __init__(self, model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH, device: str = "cuda"):
        from transformers import AutoProcessor

        model_cls = _resolve_qwen_vl_model_class()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else device,
            trust_remote_code=True,
        )

    def process_messages(self, messages, max_new_tokens=1024, do_sample=False, temperature=0.0):
        from qwen_vl_utils import process_vision_info

        qwen_messages = _to_qwen_messages(messages)
        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def _resolve_qwen_vl_model_class():
    import transformers

    for name in (
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
    ):
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
    raise ImportError("No compatible Qwen-VL model class found in transformers.")


def get_local_qwen_client(model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH, device: str = "cuda"):
    cache_key = (model_path, device)
    if not hasattr(get_local_qwen_client, "_cache"):
        get_local_qwen_client._cache = {}
    cache = get_local_qwen_client._cache
    if cache_key not in cache:
        cache[cache_key] = LocalQwenClient(model_path=model_path, device=device)
    return cache[cache_key]


def generate_text_with_local_qwen(prompt: str, model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH, device: str = "cuda", max_new_tokens: int = 1024):
    client = get_local_qwen_client(model_path=model_path, device=device)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return client.process_messages(messages, max_new_tokens=max_new_tokens)


def answer_with_local_qwen(messages, model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH, device: str = "cuda", max_new_tokens: int = 1024):
    client = get_local_qwen_client(model_path=model_path, device=device)
    prompt = (
        "Return the final response as JSON with keys reasoning and answer. "
        "The answer value must be the open-form final answer, for example yes, no, or a number."
    )
    qwen_messages = list(messages)
    qwen_messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return client.process_messages(qwen_messages, max_new_tokens=max_new_tokens)


def _extract_json_object_text(text: str) -> Optional[str]:
    value = str(text or "").strip()
    if not value:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start = value.find("{")
    end = value.rfind("}")
    if start != -1 and end != -1 and end > start:
        return value[start:end + 1].strip()
    return value


def parse_spatial_answer_text(text: str):
    from agent.anwer import SpatialAnswer

    for candidate in (str(text or "").strip(), _extract_json_object_text(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and ("answer" in data or "reasoning" in data):
                return SpatialAnswer(
                    reasoning=str(data.get("reasoning", "")),
                    answer=str(data.get("answer", "")).strip(),
                )
        except Exception:
            pass

    return SpatialAnswer(reasoning=str(text or "").strip(), answer=str(text or "").strip())


def _to_qwen_messages(messages):
    output = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            output.append({"role": message.get("role", "user"), "content": [{"type": "text", "text": content}]})
            continue
        qwen_content = []
        for item in content:
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                qwen_content.append({"type": "text", "text": item.get("text", "")})
            elif item_type in {"image", "input_image"}:
                image = item.get("image")
                image_url = item.get("image_url")
                if image is not None:
                    qwen_content.append({"type": "image", "image": image})
                elif image_url:
                    qwen_content.append({"type": "image", "image": _image_from_url_or_path(image_url)})
        output.append({"role": message.get("role", "user"), "content": qwen_content})
    return output


def _image_from_url_or_path(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str) and value.startswith("data:image"):
        _, b64 = value.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    if isinstance(value, str) and os.path.exists(value):
        return Image.open(value).convert("RGB")
    return value
