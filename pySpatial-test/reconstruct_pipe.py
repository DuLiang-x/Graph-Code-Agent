#!/usr/bin/env python3

import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import trimesh
from datetime import datetime
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from PIL import Image, ImageDraw
import base64
import io

# Support both top-level 'vggt' install and repo-local path
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
except ModuleNotFoundError:
    import sys as _sys, os as _os
    _repo_root = _os.path.dirname(_os.path.abspath(__file__))
    _vggt_root = _os.path.join(_repo_root, 'base_model', 'vggt')
    if _vggt_root not in _sys.path:
        _sys.path.insert(0, _vggt_root)
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues


# ============================================================================
# HTML Visualization Utility Functions (following VADAR pattern)
# ============================================================================

def depth_to_grayscale(depth_map, normalize=True):
    """
    Convert depth map to grayscale PIL image for visualization.
    Following VADAR pattern: normalize float depth to uint8.

    Args:
        depth_map: HxW depth map array (numpy or tensor)
        normalize: Whether to normalize depth values to [0, 255]

    Returns:
        PIL Image in grayscale mode
    """
    # Convert to numpy if it's a tensor
    if hasattr(depth_map, 'cpu'):
        depth_map = depth_map.cpu().numpy()
    
    # Ensure it's a numpy array
    depth_map = np.array(depth_map, dtype=np.float32)
    
    # Handle any extra dimensions by squeezing
    if depth_map.ndim == 3:
        # Try to squeeze channel dimension
        if depth_map.shape[-1] == 1:
            depth_map = depth_map.squeeze(axis=-1)
        elif depth_map.shape[0] == 1:
            depth_map = depth_map.squeeze(axis=0)
        else:
            # Take first channel if multiple channels
            depth_map = depth_map[..., 0]
    
    # Validate we have 2D array now
    if depth_map.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got {depth_map.ndim}D with shape {depth_map.shape}")
    
    # Get valid finite values for normalization
    depth_valid = depth_map[np.isfinite(depth_map)]
    if len(depth_valid) == 0:
        return Image.new('L', (depth_map.shape[1], depth_map.shape[0]), 0)

    if normalize:
        depth_min = depth_valid.min()
        depth_max = depth_valid.max()
        if depth_max > depth_min:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_normalized = np.zeros_like(depth_map)
    else:
        depth_normalized = np.clip(depth_map, 0, 255)

    depth_normalized = depth_normalized.astype(np.uint8)
    return Image.fromarray(depth_normalized, mode='L')


def box_image(img, boxes):
    """
    Draw bounding boxes on image.
    Following VADAR pattern: handle numpy arrays and PIL Images flexibly.

    Args:
        img: PIL Image or numpy array
        boxes: List of [x1, y1, x2, y2] bounding boxes

    Returns:
        PIL Image with boxes drawn
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Handle float mode images (depth maps)
    if img.mode == 'F':
        img = depth_to_grayscale(np.array(img))
        img = img.convert('RGB')
    elif img.mode == 'L':
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    return img


def dotted_image(img, points):
    """
    Draw dots on image at specified points.
    Following VADAR pattern: handle numpy arrays and PIL Images flexibly.

    Args:
        img: PIL Image or numpy array
        points: List of [x, y] coordinates

    Returns:
        PIL Image with dots drawn
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Handle float mode images (depth maps)
    if img.mode == 'F':
        img = depth_to_grayscale(np.array(img))
        img = img.convert('RGB')
    elif img.mode == 'L':
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img)
    img_width = img.width
    dot_size = max(2, int(img_width * 0.02))  # 2% of image width
    
    for point in points:
        x, y = point
        # Draw red dot with black outline
        draw.ellipse([x - dot_size, y - dot_size, x + dot_size, y + dot_size],
                    fill='red', outline='black')
    return img


def html_embed_image(img, size=640):
    """
    Convert PIL image to base64-encoded JPEG data URI for HTML embedding.
    Following VADAR pattern: handle float mode ('F') by converting to RGB.

    Args:
        img: PIL Image or numpy array
        size: Maximum size for the embedded image (maintains aspect ratio)

    Returns:
        HTML img tag with base64-encoded image data
    """
    if isinstance(img, np.ndarray):
        # Convert numpy array to PIL Image
        img = Image.fromarray(img)
    
    # Handle PIL Image with float mode (common for depth maps)
    if img.mode == 'F':
        img = img.convert('RGB')
    elif img.mode == 'L':
        # Grayscale to RGB for JPEG embedding
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        # RGBA to RGB for JPEG
        img = img.convert('RGB')

    # Resize if needed
    if max(img.size) > size:
        ratio = size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to JPEG and encode as base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/jpeg;base64,{img_str}" style="max-width:100%; height:auto;">'


def html_image_thumbnail(img, size=300):
    """
    Create a thumbnail version of image for inline HTML embedding.
    
    Args:
        img: PIL Image or numpy array
        size: Maximum dimension for thumbnail
    
    Returns:
        HTML img tag with base64-encoded thumbnail
    """
    return html_embed_image(img, size=size)


class HTMLTraceManager:
    """
    Manages HTML trace writing and report generation for visualization.
    Following the VADAR pattern of creating self-contained HTML reports.
    """
    
    def __init__(self, output_dir, scene_name="Scene"):
        """
        Initialize HTML trace manager.
        
        Args:
            output_dir: Directory to save HTML reports
            scene_name: Name of the scene being processed
        """
        self.output_dir = output_dir
        self.scene_name = scene_name
        self.trace_content = []
        os.makedirs(output_dir, exist_ok=True)
    
    def write_trace(self, html_content):
        """
        Append HTML content to the trace.
        
        Args:
            html_content: HTML string to add to the trace
        """
        self.trace_content.append(html_content)
    
    def write_section_header(self, title, level=2):
        """
        Write a section header to the trace.
        
        Args:
            title: Section title
            level: Header level (h1, h2, h3, etc.)
        """
        self.write_trace(f'<h{level} style="color: #2c3e50; margin-top: 30px; border-bottom: 2px solid #3498db; padding-bottom: 5px;">{title}</h{level}>')
    
    def write_info_table(self, info_dict):
        """
        Write an information table to the trace.
        
        Args:
            info_dict: Dictionary with key-value pairs to display
        """
        html = '<table style="border-collapse: collapse; width: 100%; margin: 15px 0;">'
        html += '<tr style="background-color: #f8f9fa;"><th style="border: 1px solid #ddd; padding: 8px; text-align: left; width: 30%;">Property</th>'
        html += '<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th></tr>'
        
        for key, value in info_dict.items():
            bg = '#f8f9fa' if list(info_dict.keys()).index(key) % 2 == 0 else '#ffffff'
            html += f'<tr style="background-color: {bg};">'
            html += f'<td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{key}</td>'
            html += f'<td style="border: 1px solid #ddd; padding: 8px;">{value}</td>'
            html += '</tr>'
        
        html += '</table>'
        self.write_trace(html)
    
    def write_image_with_caption(self, img, caption, size=640):
        """
        Write an image with caption to the trace.
        
        Args:
            img: PIL Image or numpy array
            caption: Caption text
            size: Maximum size for the image
        """
        html = '<div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">'
        html += f'<p style="font-weight: bold; margin-bottom: 10px;">{caption}</p>'
        html += html_embed_image(img, size=size)
        html += '</div>'
        self.write_trace(html)
    
    def generate_report(self, filename="trace.html", title=None):
        """
        Generate the complete HTML report.
        
        Args:
            filename: Output filename
            title: Report title (optional)
        """
        if title is None:
            title = f"{self.scene_name} - Reconstruction Report"
        
        html_header = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
        }}
        .image-container {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        html_footer = """
</body>
</html>"""
        
        full_html = html_header + '\n'.join(self.trace_content) + html_footer
        
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"HTML report saved to: {report_path}")
        return report_path
    
    def generate_point_cloud_report(self, points_3d, points_rgb, images=None, 
                                   depth_map=None, depth_conf=None, image_names=None):
        """
        Generate a comprehensive HTML report for point cloud reconstruction.
        
        Args:
            points_3d: Nx3 array of 3D points
            points_rgb: Nx3 array of RGB colors
            images: Optional list of input images
            depth_map: Optional depth map array
            depth_conf: Optional confidence map array
            image_names: Optional list of image names
        """
        # Header section
        self.write_section_header("Reconstruction Summary")
        
        info = {
            "Scene": self.scene_name,
            "Total Points": f'<span class="success">{len(points_3d):,}</span>',
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.write_info_table(info)
        
        # Input images section
        if images is not None and len(images) > 0:
            self.write_section_header("Input Images")
            num_images_to_show = min(4, len(images))
            for i in range(num_images_to_show):
                try:
                    if isinstance(images[i], np.ndarray):
                        img = Image.fromarray(images[i])
                    else:
                        img = Image.open(images[i])

                    name = image_names[i] if image_names and i < len(image_names) else f"Image {i}"
                    self.write_image_with_caption(img, f"Input Image: {name}", size=640)
                except Exception as e:
                    self.write_trace(f'<p class="warning">Failed to load image {i}: {str(e)}</p>')
        
        # Depth maps section
        if depth_map is not None:
            self.write_section_header("Depth Maps")
            num_frames = depth_map.shape[0]
            num_to_show = min(4, num_frames)
            indices = np.linspace(0, num_frames - 1, num_to_show, dtype=int)

            for idx in indices:
                try:
                    # depth_to_grayscale handles all validation and shape conversion
                    depth_gray = depth_to_grayscale(depth_map[idx])
                    name = image_names[idx] if image_names and idx < len(image_names) else f"Frame {idx}"
                    self.write_image_with_caption(depth_gray, f"Depth Map - {name}", size=640)

                    if depth_conf is not None:
                        conf_gray = depth_to_grayscale(depth_conf[idx])
                        self.write_image_with_caption(conf_gray, f"Confidence Map - {name}", size=640)
                except Exception as e:
                    self.write_trace(f'<p class="warning">Failed to display depth/confidence map {idx}: {str(e)}</p>')
        
        # Point cloud statistics
        self.write_section_header("Point Cloud Statistics")
        
        stats = {
            "Bounding Box X": f"[{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]",
            "Bounding Box Y": f"[{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]",
            "Bounding Box Z": f"[{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]",
            "Mean RGB": f"({points_rgb[:, 0].mean():.1f}, {points_rgb[:, 1].mean():.1f}, {points_rgb[:, 2].mean():.1f})"
        }
        self.write_info_table(stats)
        
        # Generate matplotlib visualizations and embed them
        self.write_section_header("Point Cloud Distribution")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(points_3d[:, 0], bins=100, color='blue', alpha=0.7)
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Count')
        axes[0].set_title('X Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(points_3d[:, 1], bins=100, color='green', alpha=0.7)
        axes[1].set_xlabel('Y coordinate')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Y Distribution')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(points_3d[:, 2], bins=100, color='red', alpha=0.7)
        axes[2].set_xlabel('Z coordinate')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Z Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        dist_img = Image.open(buf)
        self.write_image_with_caption(dist_img, "Point Coordinate Distributions", size=1200)
        plt.close()
        plt.close('all')
        
        return self.generate_report()




class VGGTProcessor:
    def __init__(self, device="cuda", seed=42):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize model
        self.model = VGGT()
        LOCAL_MODEL_PATH = "/data/pretrain_models/model.pt"
        self.model.load_state_dict(torch.load(LOCAL_MODEL_PATH, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"VGGT Processor initialized on {self.device}")

    def process_images(self, image_paths, output_dir=None, conf_thres_value=5.0, 
                      enable_visualization=True, scene_name=None):
        """
        Process a list of image paths with VGGT and return camera poses and point cloud.

        Args:
            image_paths: List of paths to images
            output_dir: Directory to save outputs (optional)
            conf_thres_value: Confidence threshold for depth filtering
            enable_visualization: Whether to generate visualizations (default: True)
            scene_name: Name of the scene for HTML report (optional)

        Returns:
            dict: Contains camera poses, point cloud data, and metadata
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Determine scene name
        if scene_name is None:
            if isinstance(image_paths, list) and len(image_paths) > 0:
                scene_name = os.path.basename(os.path.dirname(image_paths[0]))
            else:
                scene_name = "Unknown_Scene"

        # Load and preprocess images
        vggt_resolution = 518
        img_load_resolution = 1024

        images, original_coords = load_and_preprocess_images_square(image_paths, img_load_resolution)
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)

        print(f"Processing {len(images)} images")

        # Run VGGT inference
        extrinsic, intrinsic, depth_map, depth_conf = self._run_vggt(images, vggt_resolution)

        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

        # Generate point cloud with colors
        points_3d, points_rgb = self._generate_point_cloud(
            images, depth_map, depth_conf, extrinsic, intrinsic,
            vggt_resolution, conf_thres_value
        )

        # Check for zero points and raise exception if found
        if len(points_3d) == 0:
            raise ValueError("Scene failed: Zero valid 3D points after unprojection and filtering")

        # Rescale camera matrices to original image resolution
        scaled_extrinsic, scaled_intrinsic = self._rescale_camera_matrices(
            extrinsic, intrinsic, original_coords.cpu().numpy(), vggt_resolution
        )

        # Generate visualizations
        if enable_visualization and output_dir:
            print("Generating visualizations...")
            
            # Generate traditional matplotlib visualizations
            self.visualize_depth_maps(depth_map, depth_conf, output_dir,
                                    image_names=[os.path.basename(p) for p in image_paths])
            self.visualize_point_cloud(points_3d, points_rgb, output_dir, show=False)
            self.visualize_camera_poses(scaled_extrinsic, scaled_intrinsic, output_dir,
                                       image_names=[os.path.basename(p) for p in image_paths])
            self.visualize_coordinate_systems(scaled_extrinsic, output_dir)
            
            # Generate comprehensive HTML visualization report (VADAR-style)
            print("Generating HTML visualization report...")
            self.generate_html_report(
                output_dir=output_dir,
                scene_name=scene_name,
                points_3d=points_3d,
                points_rgb=points_rgb,
                images=images,
                depth_map=depth_map,
                depth_conf=depth_conf,
                extrinsic=scaled_extrinsic,
                intrinsic=scaled_intrinsic,
                image_paths=image_paths
            )

        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_paths': image_paths,
            'num_images': len(image_paths),
            'camera_poses': {
                'extrinsic': scaled_extrinsic.tolist(),
                'intrinsic': scaled_intrinsic.tolist()
            },
            'point_cloud': None,
            'point_cloud_path': None
        }

        # Save outputs if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save camera matrices
            camera_path = os.path.join(output_dir, "camera_matrices.npz")
            np.savez(camera_path,
                    extrinsic=scaled_extrinsic,
                    intrinsic=scaled_intrinsic,
                    image_names=[os.path.basename(p) for p in image_paths])

            # Save point cloud
            point_cloud_path = os.path.join(output_dir, "points.ply")
            trimesh.PointCloud(points_3d, colors=points_rgb).export(point_cloud_path)
            results['point_cloud_path'] = point_cloud_path

            # Save metadata
            metadata_path = os.path.join(output_dir, "processing_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to {output_dir}")
            print(f"Camera matrices saved to: {camera_path}")
            print(f"Point cloud saved to: {point_cloud_path}")

        # Add point cloud data to results
        results['point_cloud'] = {
            'points': points_3d.tolist(),
            'colors': points_rgb.tolist() if points_rgb is not None else None,
            'num_points': len(points_3d)
        }

        return results

    def _run_vggt(self, images, resolution=518):
        """Run VGGT model inference"""
        images_resized = F.interpolate(images, size=(resolution, resolution),
                                     mode="bilinear", align_corners=False)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_batch = images_resized[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)

            # Predict cameras and depth
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images_batch, ps_idx)

        # Following VADAR pattern: squeeze all size-1 dimensions, then convert to numpy
        # VGGT outputs: extrinsic (1, N, 4, 4), intrinsic (1, N, 3, 3), 
        #              depth_map (1, N, H, W) or (1, N, H, W, 1), depth_conf similar
        extrinsic_out = extrinsic.squeeze().cpu().numpy()
        intrinsic_out = intrinsic.squeeze().cpu().numpy()
        depth_map_out = depth_map.squeeze().cpu().numpy()
        depth_conf_out = depth_conf.squeeze().cpu().numpy()
        
        # Ensure depth_map has shape (N, H, W)
        # After squeeze, could be (N, H, W) if batch=1, or (H, W) if also N=1
        if depth_map_out.ndim == 2:
            # Single frame case: (H, W) -> (1, H, W)
            depth_map_out = depth_map_out[np.newaxis, ...]
            depth_conf_out = depth_conf_out[np.newaxis, ...]
        elif depth_map_out.ndim != 3:
            raise ValueError(f"Invalid depth_map dimensions after squeeze: {depth_map_out.shape}, expected (N, H, W) or (H, W)")
        
        return (extrinsic_out, intrinsic_out, depth_map_out, depth_conf_out)

    def _generate_point_cloud(self, images, depth_map, depth_conf, extrinsic, intrinsic,
                             vggt_resolution, conf_thres_value):
        """Generate point cloud with RGB colors"""
        # Unproject depth to 3D points
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        
        num_frames, height, width, _ = points_3d.shape
        
        # Get RGB colors for points
        points_rgb = F.interpolate(
            images, size=(vggt_resolution, vggt_resolution), 
            mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)
        
        # Flatten points and colors for point cloud export
        points_3d_flat = points_3d.reshape(-1, 3)
        points_rgb_flat = points_rgb.reshape(-1, 3)
        
        # Remove invalid points (NaN or infinite values) and corresponding colors
        valid_mask = np.isfinite(points_3d_flat).all(axis=1)
        points_3d_filtered = points_3d_flat[valid_mask]
        points_rgb_filtered = points_rgb_flat[valid_mask]
        
        return points_3d_filtered, points_rgb_filtered

    def _rescale_camera_matrices(self, extrinsic, intrinsic, original_coords, img_size):
        """Rescale camera matrices to original image coordinates"""
        scaled_intrinsic = intrinsic.copy()

        for i in range(len(extrinsic)):
            real_image_size = original_coords[i, -2:]
            resize_ratio = max(real_image_size) / img_size

            # Scale focal lengths and principal point
            scaled_intrinsic[i, :2, :] *= resize_ratio
            # Set principal point to image center
            scaled_intrinsic[i, 0, 2] = real_image_size[0] / 2
            scaled_intrinsic[i, 1, 2] = real_image_size[1] / 2

        return extrinsic, scaled_intrinsic

    def visualize_point_cloud(self, points_3d, points_rgb, output_dir, show=False):
        """
        Visualize point cloud using Open3D and save interactive visualization.
        
        Args:
            points_3d: Nx3 array of 3D points
            points_rgb: Nx3 array of RGB colors (0-255)
            output_dir: Directory to save visualization
            show: Whether to show interactive window (requires display)
        """
        if len(points_3d) == 0:
            print("No points to visualize")
            return

        print("Visualizing point cloud...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(points_rgb.astype(np.float64) / 255.0)
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        
        # Save point cloud visualization
        vis_path = os.path.join(output_dir, "point_cloud_visualization.png")
        
        # Create visualization with custom viewpoint
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, visible=show)
        vis.add_geometry(pcd)
        vis.add_geometry(coordinate_frame)
        
        # Set viewpoint
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.5)
        
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(vis_path)
        
        if not show:
            vis.destroy_window()
        
        print(f"Point cloud visualization saved to: {vis_path}")
        
        # Also create statistics visualization
        self._create_point_cloud_statistics(points_3d, points_rgb, output_dir)

    def _create_point_cloud_statistics(self, points_3d, points_rgb, output_dir):
        """Create statistical plots of point cloud properties"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Point distribution along X, Y, Z axes
        axes[0, 0].hist(points_3d[:, 0], bins=100, color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('X coordinate')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Point Distribution (X axis)')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(points_3d[:, 1], bins=100, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Y coordinate')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Point Distribution (Y axis)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].hist(points_3d[:, 2], bins=100, color='red', alpha=0.7)
        axes[0, 2].set_xlabel('Z coordinate')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Point Distribution (Z axis)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 2. RGB color distribution
        axes[1, 0].hist(points_rgb[:, 0], bins=50, color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Red channel')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Red Channel Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(points_rgb[:, 1], bins=50, color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Green channel')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Green Channel Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].hist(points_rgb[:, 2], bins=50, color='blue', alpha=0.7)
        axes[1, 2].set_xlabel('Blue channel')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Blue Channel Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        stats_path = os.path.join(output_dir, "point_cloud_statistics.png")
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Point cloud statistics saved to: {stats_path}")

    def visualize_camera_poses(self, extrinsics, intrinsics, output_dir, image_names=None):
        """
        Visualize camera positions and orientations in 3D space.
        
        Args:
            extrinsics: Nx4x4 or Nx3x4 camera extrinsic matrices
            intrinsics: Nx3x3 camera intrinsic matrices
            output_dir: Directory to save visualization
            image_names: Optional list of image names for labeling
        """
        print("Visualizing camera poses...")
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        num_cameras = len(extrinsics)
        
        # Extract camera centers and plot trajectories
        camera_centers = []
        for i in range(num_cameras):
            # Extract camera center from extrinsic matrix
            if extrinsics[i].shape == (3, 4):
                # Convert 3x4 to 4x4 if needed
                R = extrinsics[i][:3, :3]
                t = extrinsics[i][:3, 3]
            else:
                R = extrinsics[i][:3, :3]
                t = extrinsics[i][:3, 3]
            
            # Camera center in world coordinates: C = -R^T * t
            camera_center = -R.T @ t
            camera_centers.append(camera_center)
            
            # Plot camera position
            ax.scatter(camera_center[0], camera_center[1], camera_center[2],
                      c='red', s=100, marker='o', label='Camera' if i == 0 else '')
            
            # Draw camera frustum (simplified)
            scale = 0.5
            # Camera optical axis (forward direction)
            forward = R[:, 2]  # Third column of rotation matrix
            ax.quiver(camera_center[0], camera_center[1], camera_center[2],
                     forward[0]*scale, forward[1]*scale, forward[2]*scale,
                     color='blue', alpha=0.6, linewidth=2, label='Forward' if i == 0 else '')
            
            # Add label
            if image_names and i < len(image_names):
                label = os.path.basename(image_names[i])
            else:
                label = f'Cam {i}'
            ax.text(camera_center[0], camera_center[1], camera_center[2],
                   label, fontsize=8)
        
        camera_centers = np.array(camera_centers)
        
        # Plot camera trajectory
        if num_cameras > 1:
            ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
                   'g--', alpha=0.5, linewidth=2, label='Camera trajectory')
        
        # Draw coordinate axes
        origin = np.array([0, 0, 0])
        axis_length = np.max(np.abs(camera_centers)) * 0.1
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', alpha=0.3, label='X axis')
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', alpha=0.3, label='Y axis')
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', alpha=0.3, label='Z axis')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses and Trajectory')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        max_range = np.max(np.abs(camera_centers)) * 1.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        camera_viz_path = os.path.join(output_dir, "camera_poses_visualization.png")
        plt.savefig(camera_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Camera poses visualization saved to: {camera_viz_path}")
        
        # Also save camera path statistics
        self._create_camera_statistics(camera_centers, output_dir)

    def _create_camera_statistics(self, camera_centers, output_dir):
        """Create statistical plots of camera positions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Camera positions over time (index)
        ax = axes[0, 0]
        ax.plot(range(len(camera_centers)), camera_centers[:, 0], 'r-', label='X', alpha=0.7)
        ax.plot(range(len(camera_centers)), camera_centers[:, 1], 'g-', label='Y', alpha=0.7)
        ax.plot(range(len(camera_centers)), camera_centers[:, 2], 'b-', label='Z', alpha=0.7)
        ax.set_xlabel('Camera Index')
        ax.set_ylabel('Position')
        ax.set_title('Camera Positions Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D trajectory (top-down view)
        ax = axes[0, 1]
        ax.plot(camera_centers[:, 0], camera_centers[:, 1], 'b-', alpha=0.7, linewidth=2)
        ax.scatter(camera_centers[0, 0], camera_centers[0, 1], c='green', s=100, 
                  marker='o', label='Start', zorder=5)
        ax.scatter(camera_centers[-1, 0], camera_centers[-1, 1], c='red', s=100, 
                  marker='x', label='End', zorder=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Camera Trajectory (Top-down view)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Height distribution
        ax = axes[1, 0]
        ax.hist(camera_centers[:, 2], bins=30, color='purple', alpha=0.7)
        ax.set_xlabel('Z (Height)')
        ax.set_ylabel('Count')
        ax.set_title('Camera Height Distribution')
        ax.grid(True, alpha=0.3)
        
        # Distance between consecutive cameras
        if len(camera_centers) > 1:
            distances = np.sqrt(np.sum(np.diff(camera_centers, axis=0)**2, axis=1))
            ax = axes[1, 1]
            ax.plot(range(len(distances)), distances, 'orange', marker='o', alpha=0.7)
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Distance (m)')
            ax.set_title('Distance Between Consecutive Cameras')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        camera_stats_path = os.path.join(output_dir, "camera_statistics.png")
        plt.savefig(camera_stats_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Camera statistics saved to: {camera_stats_path}")

    def visualize_depth_maps(self, depth_map, depth_conf, output_dir, image_names=None):
        """
        Visualize depth maps and confidence maps.
        
        Args:
            depth_map: NxHxW depth map array
            depth_conf: NxHxW confidence map array
            output_dir: Directory to save visualization
            image_names: Optional list of image names for labeling
        """
        print("Visualizing depth maps...")
        
        num_frames = depth_map.shape[0]
        # Show at most 4 depth maps to avoid clutter
        num_to_show = min(4, num_frames)
        indices = np.linspace(0, num_frames - 1, num_to_show, dtype=int)
        
        fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 5 * num_to_show))
        if num_to_show == 1:
            axes = axes[np.newaxis, :]
        
        for i, idx in enumerate(indices):
            # Original depth map
            im0 = axes[i, 0].imshow(depth_map[idx], cmap='viridis')
            axes[i, 0].set_title(f'Depth Map - Frame {idx}')
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            
            # Confidence map
            im1 = axes[i, 1].imshow(depth_conf[idx], cmap='hot')
            axes[i, 1].set_title(f'Confidence Map - Frame {idx}')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            # Depth histogram
            valid_depth = depth_map[idx][np.isfinite(depth_map[idx])]
            axes[i, 2].hist(valid_depth.flatten(), bins=100, color='blue', alpha=0.7)
            axes[i, 2].set_title(f'Depth Distribution - Frame {idx}')
            axes[i, 2].set_xlabel('Depth')
            axes[i, 2].set_ylabel('Count')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        depth_viz_path = os.path.join(output_dir, "depth_maps_visualization.png")
        plt.savefig(depth_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Depth maps visualization saved to: {depth_viz_path}")

    def visualize_coordinate_systems(self, extrinsics, output_dir):
        """
        Visualize coordinate systems for each camera pose.
        
        Args:
            extrinsics: Nx4x4 or Nx3x4 camera extrinsic matrices
            output_dir: Directory to save visualization
        """
        print("Visualizing coordinate systems...")
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        num_cameras = len(extrinsics)
        axis_length = 1.0
        
        for i in range(num_cameras):
            # Extract rotation matrix
            if extrinsics[i].shape == (3, 4):
                R = extrinsics[i][:3, :3]
            else:
                R = extrinsics[i][:3, :3]
            
            # Camera center
            t = extrinsics[i][:3, 3]
            camera_center = -R.T @ t
            
            # Draw camera coordinate axes
            # X axis (red)
            x_axis = R[:, 0] * axis_length
            ax.quiver(camera_center[0], camera_center[1], camera_center[2],
                     x_axis[0], x_axis[1], x_axis[2],
                     color='red', alpha=0.6, linewidth=2)
            
            # Y axis (green)
            y_axis = R[:, 1] * axis_length
            ax.quiver(camera_center[0], camera_center[1], camera_center[2],
                     y_axis[0], y_axis[1], y_axis[2],
                     color='green', alpha=0.6, linewidth=2)
            
            # Z axis (blue)
            z_axis = R[:, 2] * axis_length
            ax.quiver(camera_center[0], camera_center[1], camera_center[2],
                     z_axis[0], z_axis[1], z_axis[2],
                     color='blue', alpha=0.6, linewidth=2)
            
            # Add label
            ax.text(camera_center[0], camera_center[1], camera_center[2],
                   f'Cam {i}', fontsize=8)
        
        # Draw world coordinate axes
        origin = np.array([0, 0, 0])
        ax.quiver(0, 0, 0, 2, 0, 0, color='red', alpha=0.8, linewidth=3, label='World X')
        ax.quiver(0, 0, 0, 0, 2, 0, color='green', alpha=0.8, linewidth=3, label='World Y')
        ax.quiver(0, 0, 0, 0, 0, 2, color='blue', alpha=0.8, linewidth=3, label='World Z')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Coordinate Systems')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        coord_viz_path = os.path.join(output_dir, "coordinate_systems.png")
        plt.savefig(coord_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Coordinate systems visualization saved to: {coord_viz_path}")

    def generate_html_report(self, output_dir, scene_name, points_3d, points_rgb,
                            images=None, depth_map=None, depth_conf=None,
                            extrinsic=None, intrinsic=None, image_paths=None):
        """
        Generate comprehensive HTML visualization report following VADAR pattern.
        
        Args:
            output_dir: Directory to save the report
            scene_name: Name of the scene
            points_3d: Nx3 array of 3D points
            points_rgb: Nx3 array of RGB colors (0-255)
            images: Optional tensor of input images
            depth_map: Optional depth map array
            depth_conf: Optional confidence map array
            extrinsic: Optional camera extrinsic matrices
            intrinsic: Optional camera intrinsic matrices
            image_paths: Optional list of image paths
        """
        html_mgr = HTMLTraceManager(output_dir, scene_name)
        
        # 1. Reconstruction Summary
        html_mgr.write_section_header("Reconstruction Summary")
        
        info = {
            "Scene Name": scene_name,
            "Total 3D Points": f'<span class="success">{len(points_3d):,}</span>',
            "Number of Images": len(image_paths) if image_paths else "N/A",
            "Processing Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Bounding Box": f"[{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}] × "
                          f"[{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] × "
                          f"[{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]"
        }
        html_mgr.write_info_table(info)
        
        # 2. Input Images
        if images is not None and image_paths:
            html_mgr.write_section_header("Input Images")
            num_images = images.shape[0]
            num_to_show = min(6, num_images)

            # Convert tensor to numpy following VADAR pattern
            # images shape: (N, C, H, W) -> (N, H, W, C)
            images_np = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

            for i in range(num_to_show):
                try:
                    img_name = os.path.basename(image_paths[i]) if i < len(image_paths) else f"Image {i}"
                    img_array = images_np[i]
                    
                    # Simple validation and conversion
                    if img_array.ndim == 3 and img_array.shape[2] in [3, 4]:
                        img = Image.fromarray(img_array)
                        html_mgr.write_image_with_caption(img, f"Input: {img_name}", size=640)
                    else:
                        html_mgr.write_trace(f'<p class="warning">Image {i} has invalid shape: {img_array.shape}</p>')
                except Exception as e:
                    html_mgr.write_trace(f'<p class="warning">Failed to display image {i}: {e}</p>')
        
        # 3. Depth Maps
        if depth_map is not None:
            html_mgr.write_section_header("Depth Maps")
            num_frames = depth_map.shape[0]
            num_to_show = min(4, num_frames)
            indices = np.linspace(0, num_frames - 1, num_to_show, dtype=int)

            for idx in indices:
                try:
                    img_name = os.path.basename(image_paths[idx]) if image_paths and idx < len(image_paths) else f"Frame {idx}"

                    # Depth map - depth_to_grayscale handles validation
                    depth_gray = depth_to_grayscale(depth_map[idx])
                    html_mgr.write_image_with_caption(depth_gray, f"Depth: {img_name}", size=640)

                    # Confidence map
                    if depth_conf is not None:
                        conf_gray = depth_to_grayscale(depth_conf[idx])
                        html_mgr.write_image_with_caption(conf_gray, f"Confidence: {img_name}", size=640)
                except Exception as e:
                    html_mgr.write_trace(f'<p class="warning">Failed to display depth/confidence map {idx}: {e}</p>')
        
        # 4. Point Cloud Distributions
        html_mgr.write_section_header("Point Cloud Statistics")
        
        # Create distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(points_3d[:, 0], bins=100, color='#3498db', alpha=0.7)
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('X Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(points_3d[:, 1], bins=100, color='#2ecc71', alpha=0.7)
        axes[1].set_xlabel('Y (m)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Y Distribution')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(points_3d[:, 2], bins=100, color='#e74c3c', alpha=0.7)
        axes[2].set_xlabel('Z (m)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Z Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        dist_img = Image.open(buf)
        html_mgr.write_image_with_caption(dist_img, "Spatial Distribution of 3D Points", size=1200)
        plt.close()
        plt.close('all')
        
        # RGB Distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(points_rgb[:, 0], bins=50, color='#e74c3c', alpha=0.7)
        axes[0].set_xlabel('Red Channel')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Red Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(points_rgb[:, 1], bins=50, color='#2ecc71', alpha=0.7)
        axes[1].set_xlabel('Green Channel')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Green Distribution')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(points_rgb[:, 2], bins=50, color='#3498db', alpha=0.7)
        axes[2].set_xlabel('Blue Channel')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Blue Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        rgb_img = Image.open(buf)
        html_mgr.write_image_with_caption(rgb_img, "Color Distribution of Point Cloud", size=1200)
        plt.close()
        plt.close('all')
        
        # 5. Camera Poses and Trajectory
        if extrinsic is not None:
            html_mgr.write_section_header("Camera Poses & Trajectory")
            
            # Extract camera centers
            camera_centers = []
            for i in range(len(extrinsic)):
                if extrinsic[i].shape == (3, 4):
                    R = extrinsic[i][:3, :3]
                    t = extrinsic[i][:3, 3]
                else:
                    R = extrinsic[i][:3, :3]
                    t = extrinsic[i][:3, 3]
                
                camera_center = -R.T @ t
                camera_centers.append(camera_center)
            
            camera_centers = np.array(camera_centers)
            
            # Camera position table
            html_mgr.write_trace('<h3>Camera Positions</h3>')
            html_mgr.write_trace('<table style="border-collapse: collapse; width: 100%;">')
            html_mgr.write_trace('<tr style="background-color: #3498db; color: white;">')
            html_mgr.write_trace('<th style="border: 1px solid #ddd; padding: 8px;">Camera</th>')
            html_mgr.write_trace('<th style="border: 1px solid #ddd; padding: 8px;">X (m)</th>')
            html_mgr.write_trace('<th style="border: 1px solid #ddd; padding: 8px;">Y (m)</th>')
            html_mgr.write_trace('<th style="border: 1px solid #ddd; padding: 8px;">Z (m)</th>')
            html_mgr.write_trace('</tr>')
            
            for i, center in enumerate(camera_centers):
                bg = '#f8f9fa' if i % 2 == 0 else '#ffffff'
                img_name = os.path.basename(image_paths[i]) if image_paths and i < len(image_paths) else f"Cam {i}"
                html_mgr.write_trace(f'<tr style="background-color: {bg};">')
                html_mgr.write_trace(f'<td style="border: 1px solid #ddd; padding: 8px;">{img_name}</td>')
                html_mgr.write_trace(f'<td style="border: 1px solid #ddd; padding: 8px;">{center[0]:.3f}</td>')
                html_mgr.write_trace(f'<td style="border: 1px solid #ddd; padding: 8px;">{center[1]:.3f}</td>')
                html_mgr.write_trace(f'<td style="border: 1px solid #ddd; padding: 8px;">{center[2]:.3f}</td>')
                html_mgr.write_trace('</tr>')
            
            html_mgr.write_trace('</table>')
            
            # Camera trajectory plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 3D trajectory (top-down)
            axes[0].plot(camera_centers[:, 0], camera_centers[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            axes[0].scatter(camera_centers[0, 0], camera_centers[0, 1], c='green', s=150, 
                          marker='o', label='Start', zorder=5, edgecolors='black')
            axes[0].scatter(camera_centers[-1, 0], camera_centers[-1, 1], c='red', s=150, 
                          marker='x', label='End', zorder=5, linewidths=3)
            
            for i in range(len(camera_centers)):
                axes[0].annotate(f'{i}', (camera_centers[i, 0], camera_centers[i, 1]), 
                               fontsize=8, alpha=0.7)
            
            axes[0].set_xlabel('X (m)')
            axes[0].set_ylabel('Y (m)')
            axes[0].set_title('Camera Trajectory (Top-down View)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].axis('equal')
            
            # Height profile
            axes[1].plot(range(len(camera_centers)), camera_centers[:, 2], 'purple', 
                        linewidth=2, marker='o', markersize=6, alpha=0.7)
            axes[1].set_xlabel('Camera Index')
            axes[1].set_ylabel('Z (m)')
            axes[1].set_title('Camera Height Profile')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            traj_img = Image.open(buf)
            html_mgr.write_image_with_caption(traj_img, "Camera Trajectory Analysis", size=1200)
            plt.close()
            plt.close('all')
            
            # Consecutive distances
            if len(camera_centers) > 1:
                distances = np.sqrt(np.sum(np.diff(camera_centers, axis=0)**2, axis=1))
                html_mgr.write_trace('<h3>Camera Movement Statistics</h3>')
                dist_info = {
                    "Total Distance Traveled": f'{np.sum(distances):.2f} m',
                    "Average Step Distance": f'{np.mean(distances):.2f} m',
                    "Max Step Distance": f'{np.max(distances):.2f} m',
                    "Min Step Distance": f'{np.min(distances):.2f} m'
                }
                html_mgr.write_info_table(dist_info)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(range(len(distances)), distances, 'orange', linewidth=2, marker='o', alpha=0.7)
                ax.set_xlabel('Frame Transition')
                ax.set_ylabel('Distance (m)')
                ax.set_title('Distance Between Consecutive Cameras')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                dist_plot = Image.open(buf)
                html_mgr.write_image_with_caption(dist_plot, "Inter-Camera Distances", size=1000)
                plt.close()
                plt.close('all')
        
        # Generate final report
        report_path = html_mgr.generate_report("reconstruction_report.html")
        print(f"Comprehensive HTML report saved to: {report_path}")
        
        return report_path



def process_jsonl_entry(processor, entry, base_data_path, base_output_path):
    """Process a single JSONL entry"""
    entry_id = entry['id']
    image_paths = entry['images']

    # Convert relative paths to absolute paths
    full_image_paths = []
    for img_path in image_paths:
        full_path = os.path.join(base_data_path, img_path)
        if os.path.exists(full_path):
            full_image_paths.append(full_path)
        else:
            print(f"Warning: Image not found: {full_path}")

    if not full_image_paths:
        print(f"No valid images found for entry {entry_id}")
        return {'entry_id': entry_id, 'status': 'failed', 'reason': 'No valid images found'}

    # Create output directory for this entry
    output_dir = os.path.join(base_output_path, entry_id)

    try:
        print(f"Processing entry {entry_id} with {len(full_image_paths)} images")
        results = processor.process_images(
            full_image_paths, 
            output_dir,
            scene_name=entry_id  # Pass scene_name for HTML report
        )
        results['entry_id'] = entry_id
        results['original_entry'] = entry
        results['status'] = 'success'
        return results
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing entry {entry_id}: {error_msg}")
        # Mark as failed with specific reason
        return {
            'entry_id': entry_id,
            'status': 'failed',
            'reason': error_msg,
            'original_entry': entry
        }


def worker_thread(gpu_id, task_queue, result_queue, base_data_path, base_output_path, conf_thres_value, seed):
    """Worker thread function that processes entries on a specific GPU"""
    device = f"cuda:{gpu_id}"
    print(f"Worker thread started on {device}")

    try:
        processor = VGGTProcessor(device=device, seed=seed + gpu_id)
    except Exception as e:
        print(f"Failed to initialize processor on {device}: {e}")
        return

    while True:
        try:
            entry = task_queue.get(timeout=1)
            if entry is None:
                break

            print(f"[GPU {gpu_id}] Processing entry: {entry['id']}")
            result = process_jsonl_entry(processor, entry, base_data_path, base_output_path)
            result['gpu_id'] = gpu_id
            result_queue.put(result)
            task_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing entry: {e}")
            if 'entry' in locals():
                error_result = {
                    'entry_id': entry.get('id', 'unknown'),
                    'status': 'failed',
                    'reason': str(e),
                    'gpu_id': gpu_id,
                    'original_entry': entry
                }
                result_queue.put(error_result)
            task_queue.task_done()


def process_batch_multithreaded(jsonl_path, base_data_path, base_output_path,
                               gpu_ids, conf_thres_value, seed, max_entries=None):
    """Process JSONL entries using multiple threads on different GPUs"""

    if not gpu_ids:
        gpu_ids = [0]

    print(f"Starting multi-threaded processing on GPUs: {gpu_ids}")

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    with open(jsonl_path, 'r') as f:
        entries_loaded = 0
        for line_num, line in enumerate(f, 1):
            if max_entries and entries_loaded >= max_entries:
                break

            try:
                entry = json.loads(line.strip())
                task_queue.put(entry)
                entries_loaded += 1
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}")

    total_entries = entries_loaded
    print(f"Loaded {total_entries} entries for processing")

    threads = []
    for gpu_id in gpu_ids:
        for _ in range(gpu_ids.count(gpu_id)):
            thread = threading.Thread(
                target=worker_thread,
                args=(gpu_id, task_queue, result_queue, base_data_path,
                     base_output_path, conf_thres_value, seed)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

    print(f"Started {len(threads)} worker threads")

    processed_count = 0
    successful_count = 0
    failed_results = []

    # Initialize progress bar
    pbar = tqdm(total=total_entries, desc="Processing entries", unit="entry")

    while processed_count < total_entries:
        try:
            result = result_queue.get(timeout=30)
            processed_count += 1

            if result['status'] == 'success':
                successful_count += 1
                pbar.set_postfix({
                    'Success': successful_count,
                    'Failed': processed_count - successful_count,
                    f'GPU{result["gpu_id"]}': f'{result["point_cloud"]["num_points"]} pts'
                })
                pbar.write(f"[GPU {result['gpu_id']}] ✓ {result['entry_id']} - "
                          f"{result['point_cloud']['num_points']} points")
            else:
                failed_results.append(result)
                pbar.set_postfix({
                    'Success': successful_count,
                    'Failed': processed_count - successful_count,
                    f'GPU{result["gpu_id"]}': 'Failed'
                })
                pbar.write(f"[GPU {result['gpu_id']}] ✗ {result['entry_id']}: {result['reason']}")

            pbar.update(1)

        except queue.Empty:
            pbar.write("Warning: Timeout waiting for results")
            break

    pbar.close()

    for _ in gpu_ids:
        task_queue.put(None)

    for thread in threads:
        thread.join(timeout=5)

    return processed_count, successful_count, failed_results


def main():
    parser = argparse.ArgumentParser(description="VGGT Image Processing Pipeline")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Directory containing input images (for single directory mode)")
    parser.add_argument("--jsonl_path", type=str, default=None,
                       help="Path to JSONL file for batch processing")
    parser.add_argument("--output_dir", type=str, default="output/preprocessed",
                       help="Directory to save outputs")
    parser.add_argument("--scene_name", type=str, default=None,
                       help="Scene name for auto output directory generation")
    parser.add_argument("--base_data_path", type=str, default=None,
                       help="Base path to resolve relative image paths in JSONL entries")
    parser.add_argument("--mode", type=str, choices=['single', 'batch'], default='batch',
                       help="Processing mode: single directory or batch from JSONL")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--conf_thres_value", type=float, default=0.0,
                       help="Confidence threshold for depth filtering")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of JSONL entries to process (for testing)")
    parser.add_argument("--gpu_ids", type=str, default="0,7",
                       help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--num_threads", type=int, default=1,
                       help="Number of threads per GPU (default: 1)")
    parser.add_argument("--multithreaded", default=False, action="store_true",
                       help="Enable multi-threaded processing across multiple GPUs")

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    # Validate GPU availability
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                print(f"Warning: GPU {gpu_id} not available. Available GPUs: 0-{available_gpus-1}")
                gpu_ids = [x for x in gpu_ids if x < available_gpus]
        if not gpu_ids:
            print("No valid GPUs found, falling back to GPU 0")
            gpu_ids = [0]
    else:
        print("CUDA not available, using CPU")
        gpu_ids = [0]

    print(f"Using GPUs: {gpu_ids} with {args.num_threads} threads per GPU")
    
    if args.mode == 'single':
        # Single directory processing mode
        if args.output_dir is None:
            scene_name = args.scene_name
            if scene_name is None:
                input_path = Path(args.input_dir)
                if input_path.name == "images":
                    scene_name = input_path.parent.name
                else:
                    scene_name = input_path.name
            
            base_output_dir = args.output_dir
            args.output_dir = os.path.join(base_output_dir, scene_name)
            print(f"Auto-generated output directory: {args.output_dir}")
        
        # Get image paths
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        
        if not image_paths:
            images_dir = os.path.join(args.input_dir, "images")
            if os.path.exists(images_dir):
                for ext in image_extensions:
                    image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
                    image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        if not image_paths:
            raise ValueError(f"No images found in {args.input_dir}")
        
        image_paths.sort()
        print(f"Found {len(image_paths)} images")
        
        # Initialize processor and run
        processor = VGGTProcessor(seed=args.seed)
        results = processor.process_images(
            image_paths, 
            output_dir=args.output_dir,
            conf_thres_value=args.conf_thres_value
        )
        
        print(f"Processing complete!")
        print(f"Processed {results['num_images']} images")
        print(f"Generated {results['point_cloud']['num_points']} 3D points")
    
    elif args.mode == 'batch':
        # Batch processing mode from JSONL
        if not os.path.exists(args.jsonl_path):
            raise ValueError(f"JSONL file not found: {args.jsonl_path}")

        # Set up paths
        base_output_path = args.output_dir

        # Create base output directory if it doesn't exist
        os.makedirs(base_output_path, exist_ok=True)

        base_data_path = args.base_data_path or os.path.dirname(args.jsonl_path)

        print(f"Processing JSONL file: {args.jsonl_path}")
        print(f"Base data path: {base_data_path}")
        print(f"Base output path: {base_output_path}")

        if args.multithreaded and (len(gpu_ids) > 1 or args.num_threads > 1):
            # Multi-threaded processing
            total_threads = len(gpu_ids) * args.num_threads
            print(f"Starting multi-threaded processing with {total_threads} threads across {len(gpu_ids)} GPUs")

            # Expand GPU list based on threads per GPU
            expanded_gpu_ids = []
            for gpu_id in gpu_ids:
                expanded_gpu_ids.extend([gpu_id] * args.num_threads)

            processed_count, successful_count, failed_results = process_batch_multithreaded(
                args.jsonl_path, base_data_path, base_output_path,
                expanded_gpu_ids, args.conf_thres_value, args.seed, args.max_entries
            )

        else:
            # Single-threaded processing (original behavior)
            if args.multithreaded:
                print("Multi-threading requested but only one GPU available, using single-threaded mode")

            # Use first GPU from the list
            device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
            processor = VGGTProcessor(device=device, seed=args.seed)

            # Process JSONL entries
            processed_count = 0
            successful_count = 0
            failed_results = []

            # Count total entries first for progress bar
            total_entries = 0
            with open(args.jsonl_path, 'r') as f:
                for line in f:
                    if args.max_entries and total_entries >= args.max_entries:
                        break
                    try:
                        json.loads(line.strip())
                        total_entries += 1
                    except json.JSONDecodeError:
                        continue

            # Initialize progress bar
            pbar = tqdm(total=total_entries, desc="Processing entries (single-threaded)", unit="entry")

            with open(args.jsonl_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if args.max_entries and processed_count >= args.max_entries:
                        pbar.write(f"Reached maximum entries limit: {args.max_entries}")
                        break

                    try:
                        entry = json.loads(line.strip())
                        processed_count += 1

                        pbar.set_description(f"Processing {entry['id']}")

                        result = process_jsonl_entry(
                            processor, entry, base_data_path, base_output_path
                        )

                        if result['status'] == 'success':
                            successful_count += 1
                            pbar.set_postfix({
                                'Success': successful_count,
                                'Failed': processed_count - successful_count,
                                'Points': f'{result["point_cloud"]["num_points"]}'
                            })
                            pbar.write(f"✓ Successfully processed {entry['id']} - "
                                      f"{result['point_cloud']['num_points']} 3D points")
                        else:
                            failed_results.append(result)
                            pbar.set_postfix({
                                'Success': successful_count,
                                'Failed': processed_count - successful_count,
                                'Status': 'Failed'
                            })
                            pbar.write(f"✗ Failed to process {entry['id']}: {result['reason']}")

                        pbar.update(1)

                    except json.JSONDecodeError:
                        pbar.write(f"Error: Invalid JSON on line {line_num}")
                    except Exception as e:
                        pbar.write(f"Error processing line {line_num}: {str(e)}")

            pbar.close()

        print(f"\n=== Batch Processing Complete ===")
        print(f"Total entries processed: {processed_count}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {processed_count - successful_count}")

        # Save failed entries report
        if failed_results:
            failed_report_path = os.path.join(base_output_path, "failed_entries_report.json")
            with open(failed_report_path, 'w') as f:
                json.dump(failed_results, f, indent=2)
            print(f"\nFailed entries report saved to: {failed_report_path}")
            print("\nFailure reasons:")
            for result in failed_results:
                print(f"  {result['entry_id']}: {result['reason']}")


if __name__ == "__main__":
    main()