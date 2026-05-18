"""Public API for standalone 3D object position extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import Object3DExtractionConfig
from .depth_module import DepthModule
from .detection_module import DetectionModule
from .orientation_module import OrientationModule
from .utils import bbox_wh, load_rgb_image, save_debug_visuals, validate_object_names


class Object3DLocator:
    def __init__(
        self,
        config: Optional[Union[Object3DExtractionConfig, Dict[str, Any]]] = None,
        device: str = "cuda",
        detection_module: Optional[Any] = None,
        depth_module: Optional[Any] = None,
        orientation_module: Optional[Any] = None,
    ):
        self.config = _coerce_config(config)
        self.device = device
        self.detection_module = detection_module or DetectionModule(self.config, device=device)
        self.depth_module = depth_module or DepthModule(self.config, device=device)
        self.orientation_module = orientation_module or OrientationModule(self.config, device=device)

    def extract(
        self,
        image,
        object_names: List[str],
        visualize: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Dict[str, object]]:
        image_pil = load_rgb_image(image)
        names = validate_object_names(object_names)

        results = {}  # type: Dict[str, Dict[str, object]]
        depth = None

        for object_name in names:
            box2d = None
            mask = None
            try:
                detections = self.detection_module.detect(image_pil, object_name)
                if not detections:
                    results[object_name] = {"error": f"No detection for object: {object_name}"}
                    continue

                best_detection = detections[0]
                box2d = [int(v) for v in best_detection["box2d"]]
                mask = self.detection_module.run_segmentation(image_pil, box2d)
                mask_area = int((mask > 0.5).sum())
                if mask_area < self.config.min_mask_area:
                    results[object_name] = {
                        "error": f"Segmentation mask too small: {mask_area} < {self.config.min_mask_area}",
                        "box2d": box2d,
                        "bbox_wh": bbox_wh(box2d),
                        "mask_area": mask_area,
                    }
                    continue

                if depth is None:
                    depth = self.depth_module.run_depth_estimation(image_pil)

                unprojected = self.depth_module.unproject_to_3D(image_pil, depth, mask)
                orientation = self.orientation_module.run_orientation_estimation(image_pil, box2d)
                results[object_name] = {
                    "position": unprojected["position"],
                    "orientation": orientation["orientation"],
                    "orientation_angles": orientation["orientation_angles"],
                    "box3d_size": unprojected["box3d_size"],
                    "box3d_center": unprojected["box3d_center"],
                    "box3d_min": unprojected["box3d_min"],
                    "box3d_max": unprojected["box3d_max"],
                    "bbox_wh": bbox_wh(box2d),
                    "box2d": box2d,
                    "mask_area": int(unprojected["mask_area"]),
                    "depth_mode": float(unprojected["depth_mode"]),
                    "num_points": int(unprojected["num_points"]),
                }

                if "score" in best_detection:
                    results[object_name]["score"] = float(best_detection["score"])

            except Exception as exc:
                error_result = {"error": str(exc)}  # type: Dict[str, object]
                if box2d is not None:
                    error_result["box2d"] = box2d
                    error_result["bbox_wh"] = bbox_wh(box2d)
                if mask is not None:
                    error_result["mask_area"] = int((mask > 0.5).sum())
                results[object_name] = error_result
            finally:
                if visualize and (box2d is not None or mask is not None):
                    save_debug_visuals(image_pil, object_name, box2d, mask, save_dir)

        return results


def _coerce_config(
    config: Optional[Union[Object3DExtractionConfig, Dict[str, Any]]]
) -> Object3DExtractionConfig:
    if config is None:
        return Object3DExtractionConfig()
    if isinstance(config, Object3DExtractionConfig):
        return config
    if isinstance(config, dict):
        cfg = Object3DExtractionConfig()
        for section_name, section_value in config.items():
            if not hasattr(cfg, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            target = getattr(cfg, section_name)
            if isinstance(section_value, dict):
                for key, value in section_value.items():
                    if not hasattr(target, key):
                        raise ValueError(f"Unknown config key: {section_name}.{key}")
                    setattr(target, key, value)
            else:
                setattr(cfg, section_name, section_value)
        return cfg
    raise TypeError("config must be None, Object3DExtractionConfig, or dict")
