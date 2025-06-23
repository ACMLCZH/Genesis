import os
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

class FrameImageExporter:
    @staticmethod
    def _export_frame_rgb_cam(export_dir, i_env, i_cam, camera_name, i_step, rgb):
        rgb = rgb[i_env, i_cam, ..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(f'{export_dir}/rgb_env{i_env}_{camera_name}_{i_step:03d}.png', rgb)

    @staticmethod
    def _export_frame_depth_cam(export_dir, i_env, i_cam, camera_name, i_step, depth):
        depth = depth[i_env, i_cam].cpu().numpy()
        cv2.imwrite(f'{export_dir}/depth_env{i_env}_{camera_name}_{i_step:03d}.png', depth)

    @staticmethod
    def _worker_export_frame_cam(args):
        export_dir, i_env, i_cam, camera_name, rgb, depth, i_step = args
        if rgb is not None:
            FrameImageExporter._export_frame_rgb_cam(export_dir, i_env, i_cam, camera_name, i_step, rgb)
        if depth is not None:
            FrameImageExporter._export_frame_depth_cam(export_dir, i_env, i_cam, camera_name, i_step, depth)

    def __init__(self, export_dir, depth_clip_max=100, depth_scale='log'):
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.depth_clip_max = depth_clip_max
        self.depth_scale = depth_scale

    def _normalize_depth(self, depth):
        """Normalize depth values for visualization.
        
        Args:
            depth: Tensor of depth values
            
        Returns:
            Normalized depth tensor as uint8
        """
        # Clip depth values
        depth = depth.clamp(0, self.depth_clip_max)
        
        # Apply scaling if specified
        if self.depth_scale == 'log':
            depth = torch.log(depth + 1)
            
        # Calculate min/max for each image in the batch
        depth_min = depth.amin(dim=(-3, -2), keepdim=True)
        depth_max = depth.amax(dim=(-3, -2), keepdim=True)
        
        # Normalize to 0-255 range
        return ((depth - depth_min) / (depth_max - depth_min) * 255).to(torch.uint8)
    
    def _get_camera_name(self, i_cam):
        return 'cam' + str(i_cam)

    def export_frame_all_cams(self, i_step, camera_idx=None, rgb=None, depth=None):
        """
        Export frames for all cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: RGB image tensor of shape (n_envs, n_cams, H, W, 3).
            depth: Depth tensor of shape (n_envs, n_cams, H, W).
        """
        if rgb is None and depth is None:
            print("No rgb or depth to export")
            return

        if rgb is not None:
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(0)
            assert rgb.ndim == 5, "rgb must be of shape (n_envs, n_cams, H, W, 3)"
        if depth is not None:
            if depth.ndim == 4:
                depth = depth.unsqueeze(0)
            depth = self._normalize_depth(depth)
            assert depth.ndim == 5, "depth must be of shape (n_envs, n_cams, H, W, 1)"
        
        if camera_idx is None:
            camera_idx = range(rgb.shape[1]) if rgb is not None else range(depth.shape[1])
        env_idx = range(rgb.shape[0]) if rgb is not None else range(depth.shape[0])
        args_list = [(self.export_dir, i_env, i_cam, self._get_camera_name(i_cam), rgb, depth, i_step) 
                     for i_env in env_idx for i_cam in camera_idx]
        with ThreadPoolExecutor() as executor:
            executor.map(FrameImageExporter._worker_export_frame_cam, args_list)

    def export_frame_single_cam(self, i_step, i_cam, rgb=None, depth=None):
        """
        Export frames for a single camera.

        Args:
            i_step: The current step index.
            cam_idx: The index of the camera.
            rgb: RGB image tensor of shape (n_envs, H, W, 3).
            depth: Depth tensor of shape (n_envs, H, W).
        """
        if rgb is not None:
            if isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb.copy())

            # Unsqueeze rgb to (n_envs, 1, H, W, 3) if n_envs > 0
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
            elif rgb.ndim == 3:
                rgb = rgb.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"Invalid rgb shape: {rgb.shape}")
            assert rgb.ndim == 5, "rgb must be of shape (n_envs, H, W, 3)"

        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth.copy())

            # Unsqueeze depth to (n_envs, 1, H, W, 1) if n_envs > 0
            if depth.ndim == 4:
                depth = depth.unsqueeze(1)
            elif depth.ndim == 3:
                depth = depth.unsqueeze(0).unsqueeze(0)
            elif depth.ndim == 2:
                depth = depth.unsqueeze(0).unsqueeze(0).unsqueeze(4)
            else:
                raise ValueError(f"Invalid depth shape: {depth.shape}")                    
            depth = self._normalize_depth(depth)
            assert depth.ndim == 5, "depth must be of shape (n_envs, H, W, 1)"
            
        env_idx = range(rgb.shape[0]) if rgb is not None else range(depth.shape[0])
        args_list = [(self.export_dir, i_env, 0, self._get_camera_name(i_cam), rgb, depth, i_step) 
                     for i_env in env_idx]
        with ThreadPoolExecutor() as executor:
            executor.map(FrameImageExporter._worker_export_frame_cam, args_list) 
