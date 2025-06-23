import os
import gc

import numpy as np
import genesis as gs
import torch
from genesis.repr_base import RBC
from .camera import Camera
from madrona_gs.renderer_gs import MadronaBatchRendererAdapter
import taichi as ti

class Light:
    def __init__(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._pos = pos
        self._dir = dir
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff

    @property
    def pos(self):
        return self._pos
    
    @property
    def dir(self):
        return self._dir
    
    @property
    def intensity(self):
        return self._intensity
    
    @property
    def directional(self):
        return self._directional
    
    @property
    def castshadow(self):
        return self._castshadow
    
    @property
    def cutoffRad(self):
        return np.deg2rad(self._cutoff)
    
    @property
    def cutoffDeg(self):
        return self._cutoff
    
class BatchRenderer(RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer, renderer_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._renderer_options = renderer_options
        self._rgb_torch = None
        self._depth_torch = None
        self._last_t = -1
        self._light_pos_tensor = None
        self._light_dir_tensor = None
        self._light_intensity_tensor = None
        self._light_directional_tensor = None
        self._light_castshadow_tensor = None
        self._light_cutoff_tensor = None
    
    def add_light(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._lights.append(Light(pos, dir, intensity, directional, castshadow, cutoff))
    
    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if(len(self._visualizer._cameras) == 0):
            raise ValueError("No cameras to render")
        cameras = self._visualizer._cameras
        lights = self._lights
        rigid = self._visualizer.scene.rigid_solver
        device = torch.cuda.current_device()
        n_envs = self._visualizer.scene.n_envs if self._visualizer.scene.n_envs > 0 else 1
        res = self._renderer_options.batch_render_res
        use_rasterizer = self._renderer_options.use_rasterizer

        # Cameras
        n_cameras = len(cameras)
        camera_pos = self._visualizer.camera_pos_all_envs_tensor
        camera_quat = self._visualizer.camera_quat_all_envs_tensor
        camera_fov = self._visualizer.camera_fov_tensor

        # Build taichi arrays to store light properties once.
        # If later we need to support dynamic lights, we should consider storing light properties as taichi fields in Genesis.
        n_lights = len(lights)
        light_pos = self.light_pos_tensor
        light_dir = self.light_dir_tensor
        light_intensity = self.light_intensity_tensor
        light_directional = self.light_directional_tensor
        light_castshadow = self.light_castshadow_tensor
        light_cutoff = self.light_cutoff_tensor

        self._renderer = MadronaBatchRendererAdapter(
            rigid,
            device,
            n_envs,
            n_cameras,
            n_lights,
            camera_fov,
            res[0],
            res[1],
            False, # add_cam_debug_geo
            use_rasterizer, # use_rasterizer
        )
        self._renderer.init(
            rigid,
            camera_pos,
            camera_quat,
            light_pos,
            light_dir,
            light_intensity,
            light_directional,
            light_castshadow,
            light_cutoff,
        )

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, force_render=False):
        """
        Render all cameras in the batch.
        """
        if(not force_render and self._last_t == self._visualizer.scene.t):
            return self._rgb_torch, self._depth_torch, None, None
        
        # Update last_t to current time to avoid re-rendering if the scene is not updated    
        self._last_t = self._visualizer.scene.t
        self.update_scene()

        rigid = self._visualizer.scene.rigid_solver
        camera_pos = self._visualizer.camera_pos_all_envs_tensor
        camera_quat = self._visualizer.camera_quat_all_envs_tensor
        # TODO: Control whether to render rgb, depth, segmentation, normal separately
        self._rgb_torch, self._depth_torch = self._renderer.render(
            rigid,
            camera_pos,
            camera_quat,
        )

        # Squeeze the first dimension of the output if n_envs == 0
        if self._visualizer.scene.n_envs == 0:
            self._rgb_torch = self._rgb_torch.squeeze(0)
            self._depth_torch = self._depth_torch.squeeze(0)
        return self._rgb_torch, self._depth_torch, None, None
    
    def destroy(self):
        self._lights.clear()
        self._rgb_torch = None
        self._depth_torch = None
    
    @property
    def lights(self):
        return self._lights
    
    def has_lights(self):
        return len(self._lights) > 0
    
    # shape of taichi matrix can't be 0, so we use 1 if there are no lights
    @property
    def light_pos_tensor(self):
        if self._light_pos_tensor is None:
            shapesize = len(self._lights) if self.has_lights() else 1
            self._light_pos_tensor = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(shapesize))
        light_positions = np.array([light.pos for light in self._lights])
        if self.has_lights():
            self._light_pos_tensor.from_numpy(light_positions.astype(np.float32))
        return self._light_pos_tensor
    
    @property
    def light_dir_tensor(self):
        if self._light_dir_tensor is None:
            shapesize = len(self._lights) if self.has_lights() else 1
            self._light_dir_tensor = ti.Matrix.field(n=3, m=1, dtype=ti.f32, shape=(shapesize))
        light_dirs = np.array([light.dir for light in self._lights])
        if self.has_lights():
            self._light_dir_tensor.from_numpy(light_dirs.astype(np.float32))
        return self._light_dir_tensor
    
    @property
    def light_intensity_tensor(self):
        if self._light_intensity_tensor is None:
            shapesize = len(self._lights) if self.has_lights() else 1
            self._light_intensity_tensor = ti.Matrix.field(n=1, m=1, dtype=ti.f32, shape=(shapesize))
        light_intensities = np.array([light.intensity for light in self._lights]).reshape(-1, 1)
        if self.has_lights():
            self._light_intensity_tensor.from_numpy(light_intensities.astype(np.float32))
        return self._light_intensity_tensor

    @property
    def light_directional_tensor(self):
        if self._light_directional_tensor is None:
            shapesize = len(self._lights) if self.has_lights() else 1
            self._light_directional_tensor = ti.Matrix.field(n=1, m=1, dtype=ti.i32, shape=(shapesize))
        light_directionals = np.array([light.directional for light in self._lights]).reshape(-1, 1)
        if self.has_lights():
            self._light_directional_tensor.from_numpy(light_directionals.astype(np.int32))
        return self._light_directional_tensor

    @property
    def light_castshadow_tensor(self):
        if self._light_castshadow_tensor is None:
            shapesize = len(self._lights) if self.has_lights() else 1
            self._light_castshadow_tensor = ti.Matrix.field(n=1, m=1, dtype=ti.i32, shape=(shapesize))
        light_castshadows = np.array([light.castshadow for light in self._lights]).reshape(-1, 1)
        if self.has_lights():
            self._light_castshadow_tensor.from_numpy(light_castshadows.astype(np.int32))
        return self._light_castshadow_tensor

    @property
    def light_cutoff_tensor(self):
        if self._light_cutoff_tensor is None:
            shapesize = len(self._lights) if self.has_lights() else 1
            self._light_cutoff_tensor = ti.Matrix.field(n=1, m=1, dtype=ti.f32, shape=(shapesize))
        light_cutoffs = np.array([light.cutoffRad for light in self._lights]).reshape(-1, 1)
        if self.has_lights():
            self._light_cutoff_tensor.from_numpy(light_cutoffs.astype(np.float32))
        return self._light_cutoff_tensor
