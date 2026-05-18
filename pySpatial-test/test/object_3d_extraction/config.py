"""Configuration for standalone 3D object position extraction."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


PACKAGE_TEST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get("OBJECT_3D_CHECKPOINT_DIR", PACKAGE_TEST_DIR / "src" / "checkpoints")
)


@dataclass
class GroundingDINOConfig:
    ckpt_path: str = str(
        Path(os.environ.get("GROUNDINGDINO_CKPT", DEFAULT_CHECKPOINT_DIR / "groundingdino_swint_ogc.pth"))
    )
    config_path: Optional[str] = os.environ.get("GROUNDINGDINO_CONFIG")
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    num_candidates: int = 5


@dataclass
class SAMConfig:
    ckpt_path: str = str(
        Path(os.environ.get("SAM_CKPT", DEFAULT_CHECKPOINT_DIR / "sam_vit_h_4b8939.pth"))
    )
    model_type: str = "default"


@dataclass
class DepthProConfig:
    ckpt_path: str = str(
        Path(os.environ.get("DEPTH_PRO_CKPT", DEFAULT_CHECKPOINT_DIR / "depth_pro.pt"))
    )


@dataclass
class OrientationConfig:
    source_path: str = str(
        Path(os.environ.get("ORIENT_ANYTHING_SOURCE", PACKAGE_TEST_DIR / "src" / "orient_anything"))
    )
    ckpt_path: str = str(Path(os.environ.get("ORIENT_ANYTHING_CKPT", DEFAULT_CHECKPOINT_DIR)))


@dataclass
class Object3DExtractionConfig:
    detection: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    segmentation: SAMConfig = field(default_factory=SAMConfig)
    depth: DepthProConfig = field(default_factory=DepthProConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    min_mask_area: int = 10
    min_depth_points: int = 10
    depth_mode_bins: int = 128
