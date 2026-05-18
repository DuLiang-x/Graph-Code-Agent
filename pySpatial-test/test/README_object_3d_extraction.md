# Object 3D Extraction

Standalone migration of the 3D object spatial position extraction path from
KAIST-Visual-AI-Group/APC-VLM.

Source attribution:

- APC-VLM repository: https://github.com/KAIST-Visual-AI-Group/APC-VLM
- Relevant files: `apc/vision_modules/depth.py`,
  `apc/vision_modules/detection.py`, `apc/vision_modules/orientation.py`,
  `apc/apc_pipeline.py`
- APC-VLM license: Apache-2.0

This module intentionally does not migrate APC-VLM VLM QA, prompt parsing,
visual prompts, numerical prompts, rendering, or perspective change. Object
orientation is estimated with Orient-Anything and returned as a semantic 3D
direction vector.

## Minimal dependencies

The current project already lists common numeric/image dependencies such as
`numpy`, `Pillow`, `opencv-python`, `scipy`, and `open3d`. This standalone
module additionally needs these model packages at runtime:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -e /path/to/GroundingDINO
pip install -e /path/to/ml-depth-pro
pip install torch transformers huggingface_hub timm
```

GroundingDINO and DepthPro are commonly installed from source:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO

git clone https://github.com/apple/ml-depth-pro.git
pip install -e ml-depth-pro
```

## Checkpoints

By default, checkpoints are read from:

```text
/data/duliang/pySpatial-test/test/src/checkpoints/
  groundingdino_swint_ogc.pth
  sam_vit_h_4b8939.pth
  depth_pro.pt
  models--Viglong--Orient-Anything/
  models--facebook--dinov2-large/
```

Orient-Anything source is vendored under:

```text
/data/duliang/pySpatial-test/test/src/orient_anything/
```

The paths can still be overridden with environment variables:

```bash
export GROUNDINGDINO_CKPT=/path/to/groundingdino_swint_ogc.pth
export SAM_CKPT=/path/to/sam_vit_h_4b8939.pth
export DEPTH_PRO_CKPT=/path/to/depth_pro.pt
export ORIENT_ANYTHING_SOURCE=/path/to/orient_anything
export ORIENT_ANYTHING_CKPT=/path/to/checkpoints
```

Checkpoint download references:

- GroundingDINO Swin-T OGC:
  https://github.com/IDEA-Research/GroundingDINO/releases
- SAM ViT-H:
  https://github.com/facebookresearch/segment-anything#model-checkpoints
- DepthPro:
  https://github.com/apple/ml-depth-pro
- Orient-Anything:
  https://huggingface.co/Viglong/Orient-Anything
- DINOv2 processor cache:
  https://huggingface.co/facebook/dinov2-large

## Python usage

```python
from object_3d_extraction import Object3DLocator

locator = Object3DLocator(device="cuda")
result = locator.extract(
    image="demo.jpg",
    object_names=["person", "chair", "dog"],
    visualize=True,
    save_dir="outputs/debug",
)
print(result)
```

## CLI usage

Run from this repository root:

By default, the CLI reads Omni3D-Bench from `/data/datasets/Omni3D-Bench/annotations.json` and resolves images under `/data/datasets/Omni3D-Bench/images`. It extracts object names from the question text:

```bash
python test/scripts/demo_extract_3d_positions.py \
  --dataset_json /data/datasets/Omni3D-Bench/annotations.json \
  --image_root /data/datasets/Omni3D-Bench/images \
  --max_samples 10 \
  --device cuda \
  --box_threshold 0.05 \
  --text_threshold 0.05 \
  --use_vlm_refinement \
  --vlm_model_path /data/pretrain_models/Qwen/models--Qwen--Qwen2.5-VL-7B-Instruct \
  --save_dir outputs/debug
```

Each sample writes one readable JSON file and, unless `--no_visualize` is passed, debug images:

```text
outputs/debug/<sample_id>/object_3d_positions.json
outputs/debug/<sample_id>/question.png
outputs/debug/<sample_id>/scene_abstraction.png
outputs/debug/<sample_id>/<object>_box.png
outputs/debug/<sample_id>/<object>_mask.png
outputs/debug/<sample_id>/<object>_overlay.png
```

`question.png` places the source image next to the sample question.
`scene_abstraction.png` renders the final abstract 3D scene using the extracted
3D box, position, and semantic orientation vector for each valid object.

To run only one sample, use either `--sample_index` or `--sample_id`:

```bash
python test/scripts/demo_extract_3d_positions.py \
  --dataset_json /data/datasets/Omni3D-Bench/annotations.json \
  --image_root /data/datasets/Omni3D-Bench/images \
  --sample_index 0 \
  --device cuda \
  --box_threshold 0.05 \
  --text_threshold 0.05 \
  --use_vlm_refinement \
  --vlm_model_path /data/pretrain_models/Qwen/models--Qwen--Qwen2.5-VL-7B-Instruct \
  --save_dir outputs/debug
```

## Output format

Each requested object returns either a result:

```json
{
  "position": [0.0, 0.0, -2.0],
  "orientation": [0.0, 0.0, -1.0],
  "orientation_angles": [0.0, 0.0, 0.0],
  "box3d_size": [0.3, 0.8, 0.2],
  "box3d_center": [0.0, 0.0, -2.0],
  "box3d_min": [-0.15, -0.4, -2.1],
  "box3d_max": [0.15, 0.4, -1.9],
  "bbox_wh": [90, 180],
  "box2d": [10, 20, 100, 200],
  "mask_area": 12345,
  "depth_mode": 2.31,
  "num_points": 4567
}
```

or an error field when detection, segmentation, or depth filtering fails:

```json
{
  "error": "No detection for object: chair"
}
```
