"""GroundingDINO + SAM wrapper for object detection and segmentation.

Adapted from KAIST-Visual-AI-Group/APC-VLM:
apc/vision_modules/detection.py, Apache-2.0 license.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .config import Object3DExtractionConfig
from .utils import clamp_box, cxcywh_to_xyxy


class DetectionModule:
    def __init__(self, config: Object3DExtractionConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.detection_model = None  # type: Any
        self.segmentation_predictor = None  # type: Any
        self._load_detection_model()
        self._load_segmentation_model()

    def _load_detection_model(self) -> None:
        try:
            from groundingdino.util.inference import load_model
            import groundingdino.config.GroundingDINO_SwinT_OGC as default_config
        except ImportError as exc:
            raise ImportError(
                "GroundingDINO is missing. Install IDEA-Research/GroundingDINO "
                "and set GROUNDINGDINO_CKPT or config.detection.ckpt_path."
            ) from exc

        config_path = self.config.detection.config_path or default_config.__file__
        self.detection_model = load_model(config_path, self.config.detection.ckpt_path)
        self.detection_model.to(self.device)

    def _load_segmentation_model(self) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment-anything is missing. Install facebookresearch/segment-anything "
                "and set SAM_CKPT or config.segmentation.ckpt_path."
            ) from exc

        sam = sam_model_registry[self.config.segmentation.model_type](
            checkpoint=self.config.segmentation.ckpt_path
        ).to(device=self.device)
        self.segmentation_predictor = SamPredictor(sam)

    def detection_process_image(self, image: Image.Image):
        try:
            import groundingdino.datasets.transforms as GT
        except ImportError as exc:
            raise ImportError("GroundingDINO transforms are missing") from exc

        transform = GT.Compose(
            [
                GT.RandomResize([800], max_size=1333),
                GT.ToTensor(),
                GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_npy = np.asarray(image)
        image_transformed, _ = transform(image, None)
        return image_npy, image_transformed

    def run_detection(self, image_tensor: Any, category: str, max_candidates: Optional[int] = None) -> List[Dict[str, object]]:
        try:
            from groundingdino.util.inference import predict
        except ImportError as exc:
            raise ImportError("GroundingDINO predict function is missing") from exc

        boxes, scores, phrases = predict(
            model=self.detection_model,
            image=image_tensor,
            caption=category,
            box_threshold=self.config.detection.box_threshold,
            text_threshold=self.config.detection.text_threshold,
        )

        boxes_np = _to_numpy(boxes)
        scores_np = _to_numpy(scores).reshape(-1)
        phrases = list(phrases) if phrases is not None else [""] * len(scores_np)
        order = np.argsort(-scores_np)

        limit = self.config.detection.num_candidates if max_candidates is None else int(max_candidates)
        detections = []  # type: List[Dict[str, object]]
        for idx in order[: max(1, limit)]:
            detections.append(
                {
                    "box": boxes_np[int(idx)],
                    "score": float(scores_np[int(idx)]),
                    "phrase": phrases[int(idx)] if int(idx) < len(phrases) else "",
                }
            )
        return detections

    def detect(self, image: Image.Image, category: str, max_candidates: Optional[int] = None) -> List[Dict[str, object]]:
        _, image_tensor = self.detection_process_image(image)
        width, height = image.size
        detections = self.run_detection(image_tensor, category, max_candidates=max_candidates)
        for det in detections:
            det["box2d"] = cxcywh_to_xyxy(np.asarray(det["box"]), width, height)
        return detections

    def run_segmentation(self, image: Image.Image, box2d: List[int]) -> np.ndarray:
        image_npy = np.array(image)
        box_arr = np.asarray(clamp_box(box2d, image.width, image.height), dtype=np.float32)
        self.segmentation_predictor.set_image(image_npy, image_format="RGB")
        masks, scores, _ = self.segmentation_predictor.predict(box=box_arr, multimask_output=True)
        if masks is None or len(masks) == 0:
            raise ValueError("SAM returned no masks")

        masks_np = np.asarray(masks)
        if scores is not None and len(scores) == len(masks_np):
            selected_idx = int(np.argmax(np.asarray(scores)))
        else:
            areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
            selected_idx = int(np.argmax(areas))
        return masks_np[selected_idx].astype(np.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)
