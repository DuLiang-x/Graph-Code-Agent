import os
import glob
import json
import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tool.recontruct import reconstruct_3d
# from tool.segment import segment_image, segment_automatic
# from tool.estimate_depth import estimate_depth
from tool.camera_understanding import analyze_camera_trajectory
from tool.novel_view_synthesis import (
    novel_view_synthesis, rotate_right, rotate_left,
    move_forward, move_backward, turn_around,
    average_look_at_directions,
)
import re
from PIL import Image


class Reconstruction:
    def __init__(self, point_cloud, extrinsics, intrinsics):
        self.point_cloud = point_cloud
        self.extrinsics = extrinsics # list of 4 *4 numpy array
        self.intrinsics = intrinsics


class Scene:
    """Simple scene class that holds image data."""

    # Base directory for image data (can be set externally)
    IMAGE_BASE_DIR = None

    def __init__(self, path_to_images: Union[str, List[str]], question: str = "", scene_id: str = None):
        self.question = question
        self.scene_id = scene_id
        self.image_paths = self._load_images(path_to_images)
        self.images = []  # Loaded PIL Images
        self.reconstruction : Reconstruction = None
        self.code : str = None
        self.visual_clue = None
        self._load_image_data()

    def _load_images(self, path_to_images: Union[str, List[str]]) -> List[str]:
        """Load image paths from directory or list."""
        if isinstance(path_to_images, str):
            if os.path.isdir(path_to_images):
                # Load all images from directory
                image_extensions = ['*.png', '*.jpg', '*.jpeg']
                images = []
                for ext in image_extensions:
                    images.extend(glob.glob(os.path.join(path_to_images, ext)))
                return sorted(images)
            else:
                # Single image file
                return [path_to_images]
        else:
            # List of image paths
            return list(path_to_images)

    def _load_image_data(self):
        """Load actual image data from paths."""
        self.images = []
        for img_path in self.image_paths:
            try:
                # Check if path already exists as-is (absolute or relative to cwd)
                if os.path.isabs(img_path) and os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    self.images.append(img)
                    continue
                
                # Check relative to cwd
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    self.images.append(img)
                    continue

                # Try with IMAGE_BASE_DIR if set
                if Scene.IMAGE_BASE_DIR:
                    full_path = os.path.join(Scene.IMAGE_BASE_DIR, img_path)
                    if os.path.exists(full_path):
                        img = Image.open(full_path).convert('RGB')
                        self.images.append(img)
                        continue

                # Image not found
                print(f"Warning: Image not found: {img_path}")
                self.images.append(None)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                self.images.append(None)


def _load_processed_scene(processed_dir):
    """Load a previously processed scene from disk.

    Supports two layouts:
      1. reconstruct_pipe.py output: camera_matrices.npz + points.ply + processing_metadata.json
      2. ReconstructionTool output: cameras.npy + points3d.npy + metadata.json

    Returns a Reconstruction object, or None if the directory doesn't contain valid data.
    """
    if not os.path.isdir(processed_dir):
        return None

    point_cloud = None
    extrinsics = None
    intrinsics = None

    # --- Layout 1: reconstruct_pipe.py ---
    npz_path = os.path.join(processed_dir, 'camera_matrices.npz')
    ply_path = os.path.join(processed_dir, 'points.ply')
    meta_path = os.path.join(processed_dir, 'processing_metadata.json')

    if os.path.exists(ply_path) and (os.path.exists(npz_path) or os.path.exists(meta_path)):
        try:
            import trimesh
            pc = trimesh.load(ply_path)
            point_cloud = np.asarray(pc.vertices)
        except Exception:
            return None

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            extrinsics = data.get('extrinsic', None)
            intrinsics = data.get('intrinsic', None)
        elif os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            camera_poses = metadata.get('camera_poses', {})
            if 'extrinsic' in camera_poses:
                extrinsics = np.array(camera_poses['extrinsic'])
            if 'intrinsic' in camera_poses:
                intrinsics = np.array(camera_poses['intrinsic'])

        return Reconstruction(point_cloud, extrinsics, intrinsics)

    # --- Layout 2: ReconstructionTool._save_results ---
    cameras_path = os.path.join(processed_dir, 'cameras.npy')
    points_path = os.path.join(processed_dir, 'points3d.npy')

    if os.path.exists(points_path):
        point_cloud = np.load(points_path)
        if os.path.exists(cameras_path):
            extrinsics = np.load(cameras_path)
        return Reconstruction(point_cloud, extrinsics, intrinsics)

    return None


class pySpatial:
    """Simple interface for 3D vision tools."""

    # Base directory where reconstruct_pipe.py saves processed scenes
    PROCESSED_BASE_DIR = None

    @staticmethod
    def reconstruct(scene: Scene, processed_dir: str = None):
        """3D reconstruction from scene images.

        If a previously processed result exists, load it instead of re-running
        reconstruction. The lookup order is:
          1. An explicit `processed_dir` argument
          2. PROCESSED_BASE_DIR / scene.scene_id  (if scene_id is set)
          3. Fall back to running reconstruct_3d()
        """
        # --- try to load cached reconstruction ---
        recon = None

        if processed_dir:
            recon = _load_processed_scene(processed_dir)
            if recon:
                print(f"Loaded processed scene from: {processed_dir}")

        if recon is None and scene.scene_id and pySpatial.PROCESSED_BASE_DIR:
            candidate = os.path.join(pySpatial.PROCESSED_BASE_DIR, scene.scene_id)
            recon = _load_processed_scene(candidate)
            if recon:
                print(f"Loaded processed scene for scene_id '{scene.scene_id}' from: {candidate}")

        if recon is not None:
            scene.reconstruction = recon
            return recon

        # --- no cached result found, run reconstruction ---
        result = reconstruct_3d(scene.images, scene_id=scene.scene_id)

        # Convert the raw result dictionary to a Reconstruction object
        point_cloud = result.get('points', None)
        cameras = result.get('cameras', None)

        # Convert point cloud to numpy if it's a tensor
        if point_cloud is not None:
            if hasattr(point_cloud, 'cpu'):  # PyTorch tensor
                point_cloud = point_cloud.cpu().numpy()
            elif hasattr(point_cloud, 'numpy'):  # Other tensor types
                point_cloud = point_cloud.numpy()

        # Extract extrinsics and intrinsics from cameras if available
        extrinsics = None
        intrinsics = None

        if cameras is not None:
            extrinsics = cameras.cpu().numpy() if hasattr(cameras, 'cpu') else cameras

        # Also check for intrinsics in the result metadata
        metadata = result.get('metadata', {})
        if metadata and isinstance(metadata, dict):
            camera_poses = metadata.get('camera_poses', {})
            if isinstance(camera_poses, dict) and 'intrinsic' in camera_poses:
                intrinsics = np.array(camera_poses['intrinsic'])

        # Create and return Reconstruction object
        reconstruction = Reconstruction(point_cloud, extrinsics, intrinsics)

        # Store the raw result for debugging
        reconstruction._raw_result = result

        scene.reconstruction = reconstruction
        return reconstruction
    
    @staticmethod
    def describe_camera_motion(recon: Reconstruction):
        """Describe camera motion from reconstruction results.
        Args:
        """
        extrinsics = recon.extrinsics
        return analyze_camera_trajectory(extrinsics)

    @staticmethod
    def synthesize_novel_view(recon: Reconstruction, new_camera_pose, width=512, height=512, out_path=None):
        """Generate novel view synthesis from reconstruction results.
        Args:
            recon: Reconstruction object with point_cloud, extrinsics, intrinsics
            new_camera_pose: 3x4 or 4x4 extrinsic matrix for the new viewpoint
            width: output image width (default: 512)
            height: output image height (default: 512)  
            out_path: output image path (default: None, returns image object if not provided)
        Returns:
            str or image: path to the rendered image if out_path provided, otherwise image object
        """
        return novel_view_synthesis(recon, new_camera_pose, width, height, out_path)
    
    
    @staticmethod
    def _get_rotation_axis(recon):
        """Compute rotation axis from reconstruction extrinsics."""
        if recon is not None and recon.extrinsics is not None:
            extrinsics = recon.extrinsics
            # Handle (N, 3, 4) or (N, 4, 4) arrays as list of matrices
            if extrinsics.ndim == 3:
                return average_look_at_directions(extrinsics)
            # Single extrinsic — can't average, fall back
        return None

    @staticmethod
    def rotate_right(extrinsic, angle=None, recon=None):
        """Rotate camera pose to the right. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        if angle is None:
            return rotate_right(extrinsic, axis=axis)
        else:
            return rotate_right(extrinsic, angle, axis=axis)

    @staticmethod
    def rotate_left(extrinsic, angle=None, recon=None):
        """Rotate camera pose to the left. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        if angle is None:
            return rotate_left(extrinsic, axis=axis)
        else:
            return rotate_left(extrinsic, angle, axis=axis)

    @staticmethod
    def move_forward(extrinsic, distance=None):
        """Move camera pose forward, Noted that a default small step is provided"""
        if distance is None:
            return move_forward(extrinsic)
        else:
            return move_forward(extrinsic, distance)

    @staticmethod
    def move_backward(extrinsic, distance=None):
        """Move camera pose backward"""
        if distance is None:
            return move_backward(extrinsic)
        else:
            return move_backward(extrinsic, distance)

    @staticmethod
    def turn_around(extrinsic, recon=None):
        """Turn camera pose around 180 degrees. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        return turn_around(extrinsic, axis=axis)


class Agent:
    _model = None
    _processor = None
    _model_name = "/data/pretrain_models/Qwen/models--Qwen--Qwen2.5-VL-7B-Instruct"  # 可通过环境变量 QWEN_MODEL 覆盖

    def __init__(self, api_key: Optional[str] = None):
        # 忽略 api_key，因为本地模型不需要
        self._load_model()

    @classmethod
    def _load_model(cls):
        """懒加载模型和 processor（静态，只加载一次）"""
        if cls._model is None:
            model_name = os.getenv("QWEN_MODEL", cls._model_name)
            print(f"Loading Qwen2.5-VL model from {model_name}...")
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            cls._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # 可改为 torch.bfloat16 或 float16
                device_map="auto",
                trust_remote_code=True
            )
            print("Model loaded.")

    def _generate(self, messages: list, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """统一的生成方法，使用 Qwen2.5-VL 的 chat 模板"""
        from qwen_vl_utils import process_vision_info
        from transformers import GenerationConfig

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info (images/videos)
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize text and process images/videos
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        # Configure generation parameters
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self._processor.tokenizer.eos_token_id,
        )
        
        # Only set sampling parameters if temperature > 0
        if temperature > 0:
            generation_config.do_sample = True
            generation_config.temperature = temperature
            generation_config.top_p = 0.9
        else:
            generation_config.do_sample = False

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Trim input tokens and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        generated = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return generated.strip()

    def generate_code(self, scene):
        """使用本地模型生成 Python 代码，根据问题调用 pySpatial 工具"""
        # 构建系统提示和用户提示
        system_prompt = (
            "You are an AI assistant that generates Python code using the pySpatial library "
            "to answer spatial reasoning questions. "
            "The library provides: pySpatial.reconstruct(), pySpatial.describe_camera_motion(), "
            "pySpatial.synthesize_novel_view(), pySpatial.rotate_right/left, move_forward/backward, turn_around. "
            "You have access to a Scene object with .reconstruction and .images. "
            "Your code should produce a visual clue (e.g., a string, a path to a rendered image, or a numeric value) "
            "that will be used to answer the question. "
            "Output only the code in a ```python``` block, without extra commentary."
        )

        # Build user message with images for Qwen2.5-VL
        user_content = []
        for img in scene.images:
            if img is not None:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": f"Question: {scene.question}\nGenerate Python code to answer this question using pySpatial:"})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = self._generate(messages, max_new_tokens=1024)
        return response

    def parse_LLM_response(self, scene, response: str) -> str:
        """提取 Python 代码块（```python ... ```）"""
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # 如果没有标记，尝试提取任意代码块
            pattern2 = r"```(.*?)```"
            match2 = re.search(pattern2, response, re.DOTALL)
            if match2:
                code = match2.group(1).strip()
            else:
                code = response.strip()
        scene.code = code
        return code

    def execute(self, scene):
        """执行已解析的代码（复用原逻辑）"""
        from agent.codeAgent.execute import execute_code  # 假设该模块可用
        try:
            program = execute_code(scene.code)
            visual_clue = program(scene)
            return visual_clue
        except Exception as e:
            import traceback
            error_details = f"Execution failed: {str(e)}\nTraceback: {traceback.format_exc()}"
            return f"there is an error during code generation, no visual clue provided. Error: {str(e)}"

    def answer(self, scene, visual_clue):
        """使用视觉线索和问题生成最终答案"""
        system_prompt = (
            "You are an AI assistant that answers spatial reasoning questions based on a given visual clue. "
            "Answer concisely and directly with the requested information (e.g., a number, a color, a direction). "
            "If the visual clue is an image path, interpret it as an image. "
            "Do not add extra commentary."
        )
        user_prompt = (
            f"Visual clue: {visual_clue}\n"
            f"Question: {scene.question}\n"
            f"Answer:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._generate(messages, max_new_tokens=256, temperature=0.0)
        # 封装成类似原接口的返回对象
        class Answer:
            def __init__(self, answer):
                self.answer = answer
        return Answer(response)

    def basic_qa(self, scene):
        """无视觉线索的 fallback QA - 使用图像直接回答"""
        system_prompt = (
            "You are an AI assistant that answers spatial reasoning questions based solely on the given images and question. "
            "Look at the images and answer concisely. "
            "Do not add extra commentary."
        )

        # Build user message with images for Qwen2.5-VL
        user_content = []
        for img in scene.images:
            if img is not None:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": f"Question: {scene.question}\nAnswer:"})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        response = self._generate(messages, max_new_tokens=256, temperature=0.0)
        class Answer:
            def __init__(self, answer):
                self.answer = answer
        return Answer(response)

    def relaxed_qa(self, scene, visual_clue=None):
        """Relaxed QA fallback - uses both visual clue and images with a more lenient prompt"""
        system_prompt = (
            "You are an AI assistant that answers spatial reasoning questions. "
            "You have access to both the original images and a visual clue from 3D analysis. "
            "Answer the question as accurately as possible based on all available information. "
            "Answer concisely with just the final answer (e.g., a number, color, or direction). "
            "Do not add extra commentary or explanation."
        )
        
        # Build user message with images and visual clue
        user_content = []
        for img in scene.images:
            if img is not None:
                user_content.append({"type": "image", "image": img})
        
        clue_text = f"\nVisual clue from 3D analysis: {visual_clue}" if visual_clue else ""
        user_content.append({
            "type": "text", 
            "text": f"Question: {scene.question}{clue_text}\nPlease provide the most accurate answer:"
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        response = self._generate(messages, max_new_tokens=256, temperature=0.3)
        class Answer:
            def __init__(self, answer):
                self.answer = answer
        return Answer(response)