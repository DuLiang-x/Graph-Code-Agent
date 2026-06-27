import os
import glob
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# from tool.segment import segment_image, segment_automatic
# from tool.estimate_depth import estimate_depth
import re


class Reconstruction:
    def __init__(self, point_cloud, extrinsics, intrinsics):
        self.point_cloud = point_cloud
        self.extrinsics = extrinsics # list of 4 *4 numpy array
        self.intrinsics = intrinsics


class Scene:
    """Simple scene class that holds image data."""

    def __init__(self, path_to_images: Union[str, List[str]], question: str = "", scene_id: str = None):
        self.question = question
        self.scene_id = scene_id
        self.images = self._load_images(path_to_images)
        self.reconstruction : Reconstruction = None
        self.code : str = None
        self.visual_clue = None
        self.object_3d_boxes: dict = None
        self.spatial_graph = None

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


OBJECT_OUTPUT_ROOTS = {
    "auto": Path("/data/duliang/pySpatial-test/outputs/Omni3D-Bench"),
    "off": Path("/data/duliang/pySpatial-test/outputs/Omni3D-Benchnomask"),
}
LEGACY_OBJECT_OUTPUT_ROOTS = {
    "off": Path("/data/duliang/pySpatial-test/outputs/Omni3D-Bench-nomask"),
}
DEFAULT_LOCAL_QWEN_MODEL_PATH = "/data/pretrain_models/Qwen/models--Qwen--Qwen2.5-VL-7B-Instruct"


def resolve_openai_base_url(base_url: str = None) -> str:
    return base_url or os.getenv("CLOSEAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def _object_output_root(mask_fallback: str) -> Path:
    if mask_fallback not in OBJECT_OUTPUT_ROOTS:
        raise ValueError("mask_fallback must be 'auto' or 'off'")
    return OBJECT_OUTPUT_ROOTS[mask_fallback]


def _object_json_paths(scene: "Scene", mask_fallback: str, save_dir: str = None) -> List[Path]:
    if not scene.scene_id:
        return []
    paths = []
    if save_dir:
        paths.append(Path(save_dir) / scene.scene_id / "object_3d_positions.json")
    paths.append(_object_output_root(mask_fallback) / scene.scene_id / "object_3d_positions.json")
    legacy_root = LEGACY_OBJECT_OUTPUT_ROOTS.get(mask_fallback)
    if legacy_root is not None:
        paths.append(legacy_root / scene.scene_id / "object_3d_positions.json")
    return paths


def _load_object_boxes_from_json(json_path: Path, scene_id: str = None) -> Optional[Dict[str, dict]]:
    if not json_path or not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    record = data.get(scene_id) if scene_id and isinstance(data, dict) else None
    if record is None and isinstance(data, dict) and len(data) == 1:
        record = next(iter(data.values()))
    if record is None and isinstance(data, dict) and "result" in data:
        record = data
    if isinstance(record, dict):
        return record.get("result", record)
    return None


def _scene_to_sample(scene: "Scene") -> Dict[str, Any]:
    return {
        "id": scene.scene_id or "scene",
        "question": scene.question,
        "images": scene.images,
    }


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
        from tool.recontruct import reconstruct_3d

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
    def extract_objects(
        scene: Scene,
        device: str = "cuda",
        mask_fallback: str = "auto",
        use_vlm_refinement: bool = True,
        vlm_model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH,
        vlm_model=None,
        visualize: bool = True,
        save_dir: str = None,
        base_data_path: str = None,
        backend: str = "local_qwen",
        api_model: str = "gpt-4.1",
        api_key: str = None,
        base_url: str = None,
        force_extract: bool = False,
        extractor=None,
        use_vlm_object_extraction: bool = True,
    ):
        if not force_extract:
            for json_path in _object_json_paths(scene, mask_fallback, save_dir=save_dir):
                cached = _load_object_boxes_from_json(json_path, scene.scene_id)
                if cached is not None:
                    print(f"[{scene.scene_id}] Loaded cached object 3D positions from {json_path}")
                    scene.object_3d_boxes = cached
                    return cached

        from scripts.demo_extract_3d_positions import extract_objects_for_sample

        output_root = Path(save_dir) if save_dir else _object_output_root(mask_fallback)
        if extractor is not None:
            print(f"[{scene.scene_id}] Running object extraction with reusable extractor")
        record, _ = extract_objects_for_sample(
            _scene_to_sample(scene),
            output_root=output_root,
            device=device,
            mask_fallback=mask_fallback,
            use_vlm_refinement=use_vlm_refinement,
            vlm_model_path=vlm_model_path,
            vlm_model=vlm_model,
            visualize=visualize,
            base_data_path=base_data_path,
            backend=backend,
            api_model=api_model,
            api_key=api_key,
            base_url=resolve_openai_base_url(base_url),
            extractor=extractor,
            use_vlm_object_extraction=use_vlm_object_extraction,
        )
        scene.object_3d_boxes = record.get("result", {})
        return scene.object_3d_boxes

    @staticmethod
    def build_graph(scene: Scene):
        from spatial_graph import SpatialGraph

        if scene.object_3d_boxes is None:
            pySpatial.extract_objects(scene)
        scene.spatial_graph = SpatialGraph(scene.object_3d_boxes or {})
        return scene.spatial_graph

    @staticmethod
    def visualize_graph(graph, output_path, width=1200, height=700):
        from spatial_graph.visualization import visualize_graph

        return visualize_graph(graph, output_path, width=width, height=height)

    @staticmethod
    def visual_color(scene: Scene, object_name: str, choices: List[str] = None) -> str:
        boxes = getattr(scene, "object_3d_boxes", None)
        payload = boxes.get(object_name) if isinstance(boxes, dict) else None
        if not isinstance(payload, dict) or not payload.get("box2d"):
            return "unknown"
        visual_api = getattr(scene, "visual_api", None)
        if visual_api is None:
            return "unknown"
        return visual_api.classify_color(scene, object_name, choices=choices)

    @staticmethod
    def visual_color_batch(scene: Scene, object_names: List[str], choices: List[str] = None) -> Dict[str, str]:
        names = [str(name) for name in object_names]
        boxes = getattr(scene, "object_3d_boxes", None)
        result = {name: "unknown" for name in names}
        if not isinstance(boxes, dict):
            return result
        valid_names = [
            name for name in names
            if isinstance(boxes.get(name), dict) and boxes[name].get("box2d")
        ]
        if not valid_names:
            return result
        visual_api = getattr(scene, "visual_api", None)
        if visual_api is None:
            return result
        result.update(visual_api.classify_color_batch(scene, valid_names, choices=choices))
        return result

    @staticmethod
    def describe_camera_motion(recon: Reconstruction):
        """Describe camera motion from reconstruction results.
        Args:
        """
        extrinsics = recon.extrinsics
        from tool.camera_understanding import analyze_camera_trajectory

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
        from tool.novel_view_synthesis import novel_view_synthesis

        return novel_view_synthesis(recon, new_camera_pose, width, height, out_path)
    
    
    @staticmethod
    def _get_rotation_axis(recon):
        """Compute rotation axis from reconstruction extrinsics."""
        if recon is not None and recon.extrinsics is not None:
            extrinsics = recon.extrinsics
            # Handle (N, 3, 4) or (N, 4, 4) arrays as list of matrices
            if extrinsics.ndim == 3:
                from tool.novel_view_synthesis import average_look_at_directions

                return average_look_at_directions(extrinsics)
            # Single extrinsic — can't average, fall back
        return None

    @staticmethod
    def rotate_right(extrinsic, angle=None, recon=None):
        """Rotate camera pose to the right. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        if angle is None:
            from tool.novel_view_synthesis import rotate_right as _rotate_right

            return _rotate_right(extrinsic, axis=axis)
        else:
            from tool.novel_view_synthesis import rotate_right as _rotate_right

            return _rotate_right(extrinsic, angle, axis=axis)

    @staticmethod
    def rotate_left(extrinsic, angle=None, recon=None):
        """Rotate camera pose to the left. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        if angle is None:
            from tool.novel_view_synthesis import rotate_left as _rotate_left

            return _rotate_left(extrinsic, axis=axis)
        else:
            from tool.novel_view_synthesis import rotate_left as _rotate_left

            return _rotate_left(extrinsic, angle, axis=axis)

    @staticmethod
    def move_forward(extrinsic, distance=None):
        """Move camera pose forward, Noted that a default small step is provided"""
        if distance is None:
            from tool.novel_view_synthesis import move_forward as _move_forward

            return _move_forward(extrinsic)
        else:
            from tool.novel_view_synthesis import move_forward as _move_forward

            return _move_forward(extrinsic, distance)

    @staticmethod
    def move_backward(extrinsic, distance=None):
        """Move camera pose backward"""
        if distance is None:
            from tool.novel_view_synthesis import move_backward as _move_backward

            return _move_backward(extrinsic)
        else:
            from tool.novel_view_synthesis import move_backward as _move_backward

            return _move_backward(extrinsic, distance)

    @staticmethod
    def turn_around(extrinsic, recon=None):
        """Turn camera pose around 180 degrees. Uses recon extrinsics to compute rotation axis."""
        axis = pySpatial._get_rotation_axis(recon)
        from tool.novel_view_synthesis import turn_around as _turn_around

        return _turn_around(extrinsic, axis=axis)


class Agent:
    def __init__(
        self,
        api_key: str = None,
        backend: str = "local_qwen",
        local_model_path: str = DEFAULT_LOCAL_QWEN_MODEL_PATH,
        model_path: str = None,
        api_model: str = "gpt-4.1",
        code_model: str = None,
        answer_model: str = None,
        base_url: str = None,
        device: str = "cuda",
    ):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.backend = backend
        self.local_model_path = model_path or local_model_path
        self.api_model = api_model
        self.code_model = code_model or api_model
        self.answer_model = answer_model or api_model
        self.base_url = resolve_openai_base_url(base_url)
        self.device = device
        
    def generate_code(self, scene: Scene):
        from agent.codeAgent.query import generate_code_from_query
        return generate_code_from_query(
            scene,
            self.api_key,
            backend=self.backend,
            model=self.code_model,
            local_model_path=self.local_model_path,
            base_url=self.base_url,
            device=self.device,
        )
        
    def repair_code(self, scene: Scene, previous_response: str = None, previous_code: str = None, error: str = None):
        from agent.codeAgent.query import repair_code_from_error
        return repair_code_from_error(
            scene,
            previous_response=previous_response,
            previous_code=previous_code,
            error=error,
            api_key=self.api_key,
            backend=self.backend,
            model=self.code_model,
            local_model_path=self.local_model_path,
            base_url=self.base_url,
            device=self.device,
        )
        
    def parse_LLM_response(self, scene: Scene, response: str):
        """
        Extracts the first python code block (```python ... ```) from text.
        Returns the code as a string, or "" if not found.
        """
        from agent.codeAgent.execute import parse_LLM_response
        code = parse_LLM_response(response)
        scene.code = code
        return code
        
    def execute(self, scene: Scene):
        """
        Execute a code string with a scene and return the visual clue result.
        """
        # try:
        #     from agent.codeAgent.execute import execute_code
        #     program = execute_code(scene.code)
            
        #     visual_clue = program(scene)
        #     return visual_clue
        # except Exception as e:
        #     import traceback
        #     error_details = f"Execution failed: {str(e)}\nTraceback: {traceback.format_exc()}"
        #     # Store the error for detailed reporting
        #     self.last_execution_error = error_details
        #     return f"there is an error during code generation, no visual clue provided. Error: {str(e)}"
        
        from agent.codeAgent.execute import execute_code
        program = execute_code(scene.code)
        from agent.visual_api import VisualAttributeClient
        scene.visual_api = VisualAttributeClient(
            backend=self.backend,
            api_key=self.api_key,
            model=self.answer_model,
            local_model_path=self.local_model_path,
            base_url=self.base_url,
            device=self.device,
        )
        
        visual_clue = program(scene)
        return visual_clue
    
    def answer(self, scene: Scene, visual_clue):
        # answer the question with visual clue
        from agent.anwer import answer

        # Set the visual clue in the scene
        scene.visual_clue = visual_clue

        # Call the answer function with API key
        return answer(scene, self.api_key, backend=self.backend, model=self.answer_model, local_model_path=self.local_model_path, base_url=self.base_url, device=self.device)

    def basic_qa(self, scene: Scene):
        """Fallback: answer using only images + question, no pySpatial framework."""
        from agent.anwer import answer_without_visual_clue
        return answer_without_visual_clue(scene, self.api_key, backend=self.backend, model=self.answer_model, local_model_path=self.local_model_path, base_url=self.base_url, device=self.device)