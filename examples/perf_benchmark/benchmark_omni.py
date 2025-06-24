# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

######################## Parse arguments #######################
import argparse
from batch_benchmark import BenchmarkArgs
args_cli = BenchmarkArgs.parse_benchmark_args()
######################## Launch app #######################
from isaaclab.app import AppLauncher
app = AppLauncher(
    headless=not args_cli.gui,
    enable_cameras=True,
    device="cuda:0",
).app

"""Rest everything follows."""
import pynvml
import torch
import os
import math
from PIL import Image
import psutil
from scipy.spatial.transform import Rotation as R
from pxr import PhysxSchema
from benchmark_profiler import BenchmarkProfiler

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils 
import isaaclab.sim as sim_utils
import isaaclab.assets as asset_utils
import isaaclab_assets.robots as asset_robots
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.sensors import (
    Camera,
    CameraCfg,
    RayCasterCamera,
    RayCasterCameraCfg,
    TiledCamera,
    TiledCameraCfg,
    patterns,
)
import carb
import omni.replicator.core as rep
from isaaclab.utils.math import (
    orthogonalize_perspective_depth,
    unproject_depth,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)
from isaaclab.utils import configclass
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.converters import (
    MjcfConverter, MjcfConverterCfg
)
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.mjcf")
import isaacsim.asset.importer.mjcf

def load_mjcf(mjcf_path):
    return MjcfConverter(
        MjcfConverterCfg(
            asset_path=mjcf_path,
            fix_base=True,
            force_usd_conversion=True
        )
    ).usd_path

def get_robot_config():
    robot_name = f"{os.path.splitext(args_cli.mjcf)[0]}_new.xml"
    robot_basename = os.path.splitext(os.path.basename(robot_name))[0]
    robot_name = os.path.join(os.path.dirname(robot_name), robot_basename, f"{robot_basename}.usd")
    robot_path = os.path.abspath(os.path.join("genesis/assets", robot_name))
    print("Robot asset:", robot_path)

    if args_cli.mjcf.endswith("g1.xml"):
        robot_cfg = asset_utils.AssetBaseCfg(
            spawn=asset_robots.unitree.G1_CFG.spawn.copy()
        )
    elif args_cli.mjcf.endswith("go2.xml"):
        robot_cfg = asset_utils.AssetBaseCfg(
            spawn=asset_robots.unitree.UNITREE_GO2_CFG.spawn.copy()
        )
    elif args_cli.mjcf.endswith("panda.xml"):
        robot_cfg = asset_utils.AssetBaseCfg(
            spawn=asset_robots.franka.FRANKA_PANDA_CFG.spawn.copy()
        )
    else:
        raise Exception(f"Invalid robot: {args_cli.mjcf}")
    robot_cfg.spawn.usd_path = robot_path
    return robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

def get_dir_light_config():
    dir_light_pos = torch.Tensor([[0.0, 0.0, 1.5]])
    dir_light_quat = quat_from_matrix(
        create_rotation_matrix_from_view(
            dir_light_pos,
            torch.Tensor([[1.0, 1.0, -2.0]]),
            stage_utils.get_stage_up_axis()))
    dir_light_pos = tuple(dir_light_pos.detach().cpu().squeeze().numpy())
    dir_light_quat = tuple(dir_light_quat.detach().cpu().squeeze().numpy())
    dir_light_cfg = asset_utils.AssetBaseCfg(
        prim_path="/World/direct_light",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, angle=45.0),
        init_state=asset_utils.AssetBaseCfg.InitialStateCfg(
            pos=dir_light_pos, rot=dir_light_quat
        )
    )
    return dir_light_cfg


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = asset_utils.AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ground",  # Each environment should have a ground 
        # prim_path="/World/ground",        # All environment shares a ground (about 2x performance)
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath("genesis/assets/urdf/plane_usd/plane.usd")
        ),
    )

    # g1/go2
    robot: asset_utils.ArticulationCfg = get_robot_config()

    # lights
    dir_light = get_dir_light_config()


def apply_benchmark_physics_settings():
    stage = stage_utils.get_current_stage()
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuMaxRigidPatchCountAttr(8388608)
    physxSceneAPI.CreateGpuMaxRigidContactCountAttr(16777216)


def apply_benchmark_carb_settings(print_changes=False):
    rep.settings.set_render_rtx_realtime()
    settings = carb.settings.get_settings()
    # Print settings before applying the settings
    if print_changes:
        print("Before settings:")
        print("Render mode:", settings.get("/rtx/rendermode"))
        print("Sample per pixel:", settings.get("/rtx/pathtracing/spp"))
        print("Total spp:", settings.get("/rtx/pathtracing/totalSpp"))
        print("Clamp spp:", settings.get("/rtx/pathtracing/clampSpp"))
        print("Max bounce:", settings.get("/rtx/pathtracing/maxBounces"))
        print("Optix Denoiser", settings.get("/rtx/pathtracing/optixDenoiser/enabled"))
        print("Shadows", settings.get("/rtx/shadows/enabled"))
        print("dlss/enabled:", settings.get("/rtx/post/dlss/enabled"))
        print("dlss/auto:", settings.get("/rtx/post/dlss/auto"))
        print("upscaling/enabled:", settings.get("/rtx/post/upscaling/enabled"))
        print("aa/denoiser/enabled:", settings.get("/rtx/post/aa/denoiser/enabled"))
        print("aa/taa/enabled:", settings.get("/rtx/post/aa/taa/enabled"))
        print("motionBlur/enabled:", settings.get("/rtx/post/motionBlur/enabled"))
        print("dof/enabled:", settings.get("/rtx/post/dof/enabled"))
        print("bloom/enabled:", settings.get("/rtx/post/bloom/enabled"))
        print("tonemap/enabled:", settings.get("/rtx/post/tonemap/enabled"))
        print("exposure/enabled:", settings.get("/rtx/post/exposure/enabled"))
        print("vsync:", settings.get("/app/window/vsync"))

    # Options: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html
    if args_cli.rasterizer:
        # carb_settings.set("/rtx/rendermode", "Hydra Storm")
        settings.set("/rtx/rendermode", "RayTracedLighting")
    else:
        settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/shadows/enabled", False)

    # Path tracing settings
    settings.set("/rtx/pathtracing/spp", args_cli.spp)
    settings.set("/rtx/pathtracing/totalSpp", args_cli.spp)
    settings.set("/rtx/pathtracing/clampSpp", args_cli.spp)
    settings.set("/rtx/pathtracing/maxBounces", args_cli.max_bounce)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)
    settings.set("/rtx/pathtracing/adaptiveSampling/enabled", False)

    # Disable DLSS & upscaling
    settings.set("/rtx-transient/dlssg/enabled", False)
    settings.set("/rtx/post/dlss/enabled", False)
    settings.set("/rtx/post/dlss/auto", False)
    settings.set("/rtx/post/upscaling/enabled", False)

    # Disable post-processing
    settings.set("/rtx/post/aa/denoiser/enabled", False)
    settings.set("/rtx/post/aa/taa/enabled", False)
    settings.set("/rtx/post/motionBlur/enabled", False)
    settings.set("/rtx/post/dof/enabled", False)
    settings.set("/rtx/post/bloom/enabled", False)
    settings.set("/rtx/post/tonemap/enabled", False)
    settings.set("/rtx/post/exposure/enabled", False)

    # Disable VSync
    settings.set("/app/window/vsync", False)

    # Print settings after applying the settings
    if print_changes:
        print("After settings:")
        print("Render mode:", settings.get("/rtx/rendermode"))
        print("Sample per pixel:", settings.get("/rtx/pathtracing/spp"))
        print("Total spp:", settings.get("/rtx/pathtracing/totalSpp"))
        print("Clamp spp:", settings.get("/rtx/pathtracing/clampSpp"))
        print("Max bounce:", settings.get("/rtx/pathtracing/maxBounces"))
        print("Optix Denoiser", settings.get("/rtx/pathtracing/optixDenoiser/enabled"))
        print("Shadows", settings.get("/rtx/shadows/enabled"))
        print("dlss/enabled:", settings.get("/rtx/post/dlss/enabled"))
        print("dlss/auto:", settings.get("/rtx/post/dlss/auto"))
        print("upscaling/enabled:", settings.get("/rtx/post/upscaling/enabled"))
        print("aa/denoiser/enabled:", settings.get("/rtx/post/aa/denoiser/enabled"))
        print("aa/taa/enabled:", settings.get("/rtx/post/aa/taa/enabled"))
        print("motionBlur/enabled:", settings.get("/rtx/post/motionBlur/enabled"))
        print("dof/enabled:", settings.get("/rtx/post/dof/enabled"))
        print("bloom/enabled:", settings.get("/rtx/post/bloom/enabled"))
        print("tonemap/enabled:", settings.get("/rtx/post/tonemap/enabled"))
        print("exposure/enabled:", settings.get("/rtx/post/exposure/enabled"))
        print("vsync:", settings.get("/app/window/vsync"))

# python3 /home/zhehuan/Desktop/hz/tmp/Genesis/examples/perf_benchmark/benchmark_omni_env.py --rasterizer --renderer omniverse --n_envs 256 --resX 128 --resY 128 --mjcf xml/franka_emika_panda/panda.xml --benchmark_result_file logs/benchmark/batch_benchmark_20250623_145352.csv --benchmark_config_file benchmark_config_smoke_test.yml

def create_scene():
    """Loads the task, sticks cameras into the config, and creates the environment."""
    # cfg = load_cfg_from_registry(task, "env_cfg_entry_point")

    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0", dt=0.01, use_fabric=False,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = RobotSceneCfg(num_envs=args_cli.n_envs, env_spacing=10.0)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    apply_benchmark_physics_settings()
    apply_benchmark_carb_settings(True)

    camera_fov = math.radians(args_cli.camera_fov)
    camera_aperture = 20.955
    camera_fol = camera_aperture / (2 * math.tan(camera_fov / 2))

    num_cameras = args_cli.n_envs
    camera_pos = torch.tensor((
        args_cli.camera_posX,
        args_cli.camera_posY,
        args_cli.camera_posZ
    )).reshape(-1, 3)
    camera_lookat = torch.tensor((
        args_cli.camera_lookatX,
        args_cli.camera_lookatY,
        args_cli.camera_lookatZ
    )).reshape(-1, 3)
    camera_quat = quat_from_matrix(
        create_rotation_matrix_from_view(
            camera_lookat, camera_pos, stage_utils.get_stage_up_axis()
        ) @ R.from_euler('z', 180, degrees=True).as_matrix()   
    )
    camera_pos = tuple(camera_pos.detach().cpu().squeeze().numpy())
    camera_quat = tuple(camera_quat.detach().cpu().squeeze().numpy())
    camera_cfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/tiled_camera",
        update_period=0,
        height=args_cli.resX,
        width=args_cli.resY,
        offset=TiledCameraCfg.OffsetCfg(
            pos=camera_pos,
            rot=camera_quat,
            convention="ros"
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=camera_fol,
            horizontal_aperture=camera_aperture,
        ),
    )
    setattr(scene_cfg, "tiled_camera", camera_cfg)
    scene = InteractiveScene(scene_cfg)
    return sim, scene


"""
System diagnosis
"""
def get_utilization_percentages(reset: bool = False, max_values: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last time reset was true."""
    if reset:
        max_values[:] = [0, 0, 0, 0]  # Reset the max values

    # CPU utilization
    cpu_usage = psutil.cpu_percent(interval=0.1)
    max_values[0] = max(max_values[0], cpu_usage)

    # RAM utilization
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)

    # GPU utilization using pynvml
    if torch.cuda.is_available():
        pynvml.nvmlInit()  # Initialize NVML
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU Utilization
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_processing_utilization_percent = gpu_utilization.gpu  # GPU core utilization
            max_values[2] = max(max_values[2], gpu_processing_utilization_percent)

            # GPU Memory Usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = memory_info.total
            gpu_memory_used = memory_info.used
            gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
            max_values[3] = max(max_values[3], gpu_memory_utilization_percent)

        pynvml.nvmlShutdown()  # Shutdown NVML after usage
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None
    return max_values


"""
Experiment
"""
def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
) -> dict:
    """Run the simulator with all cameras, and return timing analytics. Visualize if desired."""
    n_envs = args_cli.n_envs
    n_steps = args_cli.n_steps
    camera = scene["tiled_camera"]
    camera_data_types = ["rgb", "depth"]

    # Initialize timing variables
    system_utilization_analytics = get_utilization_percentages()
    print(
        f"| CPU:{system_utilization_analytics[0]}% | "
        f"RAM:{system_utilization_analytics[1]}% | "
        f"GPU Compute:{system_utilization_analytics[2]}% | "
        f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
    )
    sim.reset()
    dt = sim.get_physics_dt()
    n_warm_steps = 3
    for i in range(n_warm_steps):
        print(f"Warm up step {i}.")
        sim.step()
        camera.update(dt)
        _ = camera.data

    print("Warm up finished.")
    image_dir = os.path.splitext(args_cli.benchmark_result_file)[0]
    # exporter = FrameImageExporter(image_dir)
    # experiment_length = args_cli.n_steps + n_warm_steps

    profiler = BenchmarkProfiler(n_steps, n_envs)
    for i in range(n_steps):
        print(f"Step {i}:")
        get_utilization_percentages()
        print("Utilization percentages got!")

        # Measure the total simulation step time
        profiler.on_simulation_start()
        sim.step(render=False)
        profiler.on_rendering_start()
        sim.render()

        # Update cameras and process vision data within the simulation step
        # Loop through all camera lists and their data_types
        camera.update(dt=dt)
        print("Camera updated!")
        rgb_tiles = camera.data.output.get("rgb")
        depth_tiles = camera.data.output.get("depth")
        profiler.on_rendering_end()
        # exporter.export_frame_single_cam(i, 0, rgb=rgb_tiles, depth=depth_tiles)

        # rgb_tiles = rgb_tiles.detach().cpu().numpy()
        # for j in range(n_envs):
        #     rgb_image = rgb_tiles[j]
        #     rgb_image = Image.fromarray(rgb_image)

        #     image_name = f"rgb_{j}_env.png"
        #     image_path = os.path.join(image_dir, image_name)
        #     rgb_image.save(image_path)
        #     print("Image saved:", image_path)
        # End timing for the step

    profiler.end()
    profiler.print_summary()

    time_taken_gpu = profiler.get_total_rendering_gpu_time()
    time_taken_cpu = profiler.get_total_rendering_cpu_time()
    time_taken_per_env_gpu = profiler.get_total_rendering_gpu_time_per_env()
    time_taken_per_env_cpu = profiler.get_total_rendering_cpu_time_per_env()
    fps = profiler.get_rendering_fps()
    fps_per_env = profiler.get_rendering_fps_per_env()
    system_utilization_analytics = get_utilization_percentages()

    system_utilization_analytics = get_utilization_percentages()
    print(
        f"| CPU:{system_utilization_analytics[0]}% | "
        f"RAM:{system_utilization_analytics[1]}% | "
        f"GPU Compute:{system_utilization_analytics[2]}% | "
        f" GPU Memory: {system_utilization_analytics[3]:.2f}% |"
    )

    os.makedirs(os.path.dirname(args_cli.benchmark_result_file), exist_ok=True)
    with open(args_cli.benchmark_result_file, 'a') as f:
        f.write(f'succeeded,{args_cli.mjcf},{args_cli.renderer},'
                f'{args_cli.rasterizer},{args_cli.n_envs},{args_cli.n_steps},'\
                f'{args_cli.resX},{args_cli.resY},'
                f'{args_cli.camera_posX},{args_cli.camera_posY},{args_cli.camera_posZ},'
                f'{args_cli.camera_lookatX},{args_cli.camera_lookatY},{args_cli.camera_lookatZ},'
                f'{args_cli.camera_fov},'
                f'{time_taken_gpu},{time_taken_per_env_gpu},{time_taken_cpu},'
                f'{time_taken_per_env_cpu},{fps},{fps_per_env}\n')
        
    print("App closing..")
    # app.close()
    print("App closed!")


def main():
    """Main function."""
    # Load simulation context
    print("[INFO]: Designing the scene")
    print("[INFO]: Using known task environment, injecting cameras.")
    sim, scene = create_scene()
    run_simulator(sim=sim, scene=scene)
    print("[INFO]: DONE! Feel free to CTRL + C Me ")
    print("Keep in mind, this is without any training running on the GPU.")
    print("Set lower utilization thresholds to account for training.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # simulation_app.close()
