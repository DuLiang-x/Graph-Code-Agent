"""DepthPro wrapper and APC-VLM-style 3D unprojection.

Portions are adapted from KAIST-Visual-AI-Group/APC-VLM:
apc/vision_modules/depth.py, Apache-2.0 license.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from .config import Object3DExtractionConfig


class DepthModule:
    def __init__(self, config: Object3DExtractionConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.depth_model = None  # type: Any
        self.depth_transform = None  # type: Any
        self._load_depth_model()

    def _load_depth_model(self) -> None:
        try:
            import torch
            import depth_pro
            from depth_pro.depth_pro import DepthProConfig
        except ImportError as exc:
            raise ImportError(
                "DepthPro dependencies are missing. Install apple/ml-depth-pro "
                "and set DEPTH_PRO_CKPT or config.depth.ckpt_path."
            ) from exc

        precision = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._depth_pro = depth_pro
        self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
            config=DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=self.config.depth.ckpt_path,
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            ),
            device=torch.device(self.device),
            precision=precision,
        )
        self.depth_model.eval()

    def depth_process_image(
        self,
        image: Image.Image,
        auto_rotate: bool = True,
        remove_alpha: bool = True,
    ):
        depth_pro = self._depth_pro
        img_exif = depth_pro.utils.extract_exif(image)
        icc_profile = image.info.get("icc_profile", None)

        if auto_rotate:
            exif_orientation = img_exif.get("Orientation", 1)
            if exif_orientation == 3:
                image = image.transpose(Image.ROTATE_180)
            elif exif_orientation == 6:
                image = image.transpose(Image.ROTATE_270)
            elif exif_orientation == 8:
                image = image.transpose(Image.ROTATE_90)

        image_npy = np.array(image)
        if image_npy.ndim < 3 or image_npy.shape[2] == 1:
            image_npy = np.dstack((image_npy, image_npy, image_npy))
        if remove_alpha:
            image_npy = image_npy[:, :, :3]

        f_35mm = img_exif.get(
            "FocalLengthIn35mmFilm",
            img_exif.get("FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)),
        )
        if f_35mm is not None and f_35mm > 0:
            f_px = depth_pro.utils.fpx_from_f35(image_npy.shape[1], image_npy.shape[0], f_35mm)
        else:
            f_px = None

        return image_npy, icc_profile, f_px

    def run_depth_estimation(self, image: Image.Image) -> np.ndarray:
        image_npy, _, f_px = self.depth_process_image(image)
        image_tensor = self.depth_transform(image_npy)

        depth_pred = self.depth_model.infer(image_tensor, f_px=f_px)
        depth = depth_pred["depth"]
        return depth.cpu().numpy().astype(np.float32)

    def unproject_to_3D(
        self,
        image: Image.Image,
        depth: np.ndarray,
        segment_mask: np.ndarray,
    ) -> Dict[str, object]:
        return unproject_to_3D(
            image=image,
            depth=depth,
            segment_mask=segment_mask,
            min_depth_points=self.config.min_depth_points,
            depth_mode_bins=self.config.depth_mode_bins,
        )


def _mode_depth(depth_values: np.ndarray, bins: int = 128) -> float:
    values = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
    if values.size == 0:
        raise ValueError("No valid positive depth values inside mask")

    try:
        from scipy.stats import mode

        mode_result = mode(values, keepdims=True)
        mode_value = float(np.asarray(mode_result.mode).reshape(-1)[0])
        if np.isfinite(mode_value) and mode_value > 0:
            return mode_value
    except Exception:
        pass

    hist, bin_edges = np.histogram(values, bins=bins)
    max_idx = int(np.argmax(hist))
    return float((bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0)



def unproject_to_3D(
    image: Image.Image,
    depth: np.ndarray,
    segment_mask: np.ndarray,
    min_depth_points: int = 10,
    depth_mode_bins: int = 128,
) -> Dict[str, object]:
    image_w, image_h = image.size
    depth = np.asarray(depth, dtype=np.float32)
    segment_mask = np.asarray(segment_mask)

    if depth.shape[:2] != (image_h, image_w):
        raise ValueError(
            f"Depth shape {depth.shape[:2]} does not match image size {(image_h, image_w)}"
        )
    if segment_mask.shape[:2] != (image_h, image_w):
        raise ValueError(
            f"Mask shape {segment_mask.shape[:2]} does not match image size {(image_h, image_w)}"
        )

    ys, xs = np.where(segment_mask > 0.5)
    mask_area = int(len(xs))
    if mask_area == 0:
        raise ValueError("Segmentation mask is empty")

    depth_values = depth[ys, xs]
    valid_mask = np.isfinite(depth_values) & (depth_values > 0)
    if int(valid_mask.sum()) < min_depth_points:
        raise ValueError(
            f"Not enough valid depth points inside mask: {int(valid_mask.sum())} < {min_depth_points}"
        )

    ys = ys[valid_mask]
    xs = xs[valid_mask]
    depth_values = depth_values[valid_mask]

    mode_depth = _mode_depth(depth_values, bins=depth_mode_bins)
    lower = 0.9 * mode_depth
    upper = 1.1 * mode_depth
    depth_filter = (depth_values >= lower) & (depth_values <= upper)

    if int(depth_filter.sum()) < min_depth_points:
        raise ValueError(
            f"Not enough depth points after mode filtering: {int(depth_filter.sum())} < {min_depth_points}"
        )

    ys = ys[depth_filter].astype(np.float32)
    xs = xs[depth_filter].astype(np.float32)
    zs = depth_values[depth_filter].astype(np.float32)

    focal_len_ndc = 4.0
    focal_len = focal_len_ndc * image_w
    px, py = image_w / 2.0, image_h / 2.0

    x_3d = zs * (xs - px) / focal_len
    y_3d = zs * (ys - py) / focal_len
    points = np.stack([x_3d, -y_3d, -zs], axis=1)
    position = np.median(points, axis=0).astype(float)
    box3d_min = np.min(points, axis=0).astype(float)
    box3d_max = np.max(points, axis=0).astype(float)
    box3d_size = box3d_max - box3d_min
    box3d_center = (box3d_min + box3d_max) / 2.0

    return {
        "position": [float(v) for v in position.tolist()],
        "box3d_size": [float(v) for v in box3d_size.tolist()],
        "box3d_center": [float(v) for v in box3d_center.tolist()],
        "box3d_min": [float(v) for v in box3d_min.tolist()],
        "box3d_max": [float(v) for v in box3d_max.tolist()],
        "depth_mode": float(mode_depth),
        "num_points": int(points.shape[0]),
        "mask_area": mask_area,
    }
