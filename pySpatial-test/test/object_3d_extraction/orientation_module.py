"""Orient-Anything wrapper for semantic 3D object orientation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import os
import sys

import numpy as np
from PIL import Image

from .config import Object3DExtractionConfig


class OrientationModule:
    def __init__(self, config: Object3DExtractionConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.orientation_model = None  # type: Any
        self.orientation_processor = None  # type: Any
        self._load_orientation_model()

    def _load_orientation_model(self) -> None:
        source_path = self.config.orientation.source_path
        os.environ["ORIENT_ANYTHING_CKPT"] = self.config.orientation.ckpt_path
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoImageProcessor
            from vision_tower import DINOv2_MLP
        except ImportError as exc:
            raise ImportError(
                "Orient-Anything dependencies are missing. Install torch, transformers, "
                "huggingface_hub, and the packages required by test/src/orient_anything."
            ) from exc

        self._torch = torch
        self.orientation_model = DINOv2_MLP(
            dino_mode="large",
            in_dim=1024,
            out_dim=360 + 180 + 180 + 2,
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
        ).eval()

        orientation_ckpt_path = hf_hub_download(
            repo_id="Viglong/Orient-Anything",
            filename="croplargeEX2/dino_weight.pt",
            repo_type="model",
            cache_dir=self.config.orientation.ckpt_path,
            local_files_only=True,
        )
        self.orientation_model.load_state_dict(
            torch.load(orientation_ckpt_path, map_location="cpu")
        )
        self.orientation_model.to(self.device)

        self.orientation_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large",
            cache_dir=self.config.orientation.ckpt_path,
            local_files_only=True,
        )

    def run_orientation_estimation(
        self,
        image: Image.Image,
        bbox: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        if bbox is not None:
            image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        if image.size[0] > image.size[1]:
            new_size = (512, int(512 * image.size[1] / image.size[0]))
        else:
            new_size = (int(512 * image.size[0] / image.size[1]), 512)
        image = image.resize(new_size)

        pred_angles = self._get_3angle(image)
        azimuth_deg = float(pred_angles[0])
        polar_deg = float(pred_angles[1])
        rotation_deg = float(pred_angles[2])
        direction = self.orientation_to_direction(azimuth_deg, polar_deg)

        return {
            "orientation": [float(v) for v in direction.tolist()],
            "orientation_angles": [azimuth_deg, polar_deg, rotation_deg],
        }

    def _get_3angle(self, image: Image.Image):
        torch = self._torch
        image_inputs = self.orientation_processor(images=image)
        image_inputs["pixel_values"] = torch.from_numpy(
            np.array(image_inputs["pixel_values"])
        ).to(self.device)

        with torch.no_grad():
            dino_pred = self.orientation_model(image_inputs)

        gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)
        gaus_pl_pred = torch.argmax(dino_pred[:, 360:540], dim=-1)
        gaus_ro_pred = torch.argmax(dino_pred[:, 540:720], dim=-1)

        return [
            float(gaus_ax_pred.item()),
            float((gaus_pl_pred - 90).item()),
            float((gaus_ro_pred - 90).item()),
        ]

    @staticmethod
    def orientation_to_direction(azimuth_deg: float, polar_deg: float) -> np.ndarray:
        return np.array(
            [
                -math.sin(math.radians(azimuth_deg)),
                math.sin(math.radians(polar_deg)) * math.cos(math.radians(azimuth_deg)),
                -math.cos(math.radians(polar_deg)) * math.cos(math.radians(azimuth_deg)),
            ]
        )
