from benchmark_plotter import plot_batch_benchmark
import argparse
import subprocess
import os
from datetime import datetime
import pandas as pd

class BenchmarkArgs:
    def __init__(self,
        rasterizer, n_envs, n_steps, resX, resY,
        camera_posX, camera_posY, camera_posZ,
        camera_lookatX, camera_lookatY, camera_lookatZ,
        camera_fov, mjcf, benchmark_result_file_path,
        max_bounce=2, spp=64,
    ):
        self.rasterizer = rasterizer
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.resX = resX
        self.resY = resY
        self.camera_posX = camera_posX
        self.camera_posY = camera_posY
        self.camera_posZ = camera_posZ
        self.camera_lookatX = camera_lookatX
        self.camera_lookatY = camera_lookatY
        self.camera_lookatZ = camera_lookatZ
        self.camera_fov = camera_fov
        self.mjcf = mjcf
        self.benchmark_result_file_path = benchmark_result_file_path
        self.max_bounce = max_bounce
        self.spp = spp

class BatchBenchmarkArgs:
    def __init__(self, use_full_list, continue_from, script):
        self.use_full_list = use_full_list
        self.continue_from = continue_from
        self.script = script

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--use_full_list", action="store_true", default=False)
    parser.add_argument("-c", "--continue_from", type=str, default=None)
    parser.add_argument("-s", "--script", type=str, default="benchmark.py")
    args = parser.parse_args()
    return BatchBenchmarkArgs(
        use_full_list=args.use_full_list,
        continue_from=args.continue_from,
        script=args.script
    )

def create_batch_args(benchmark_result_file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(benchmark_result_file_path), exist_ok=True)
    
    # Create a list of all the possible combinations of arguments
    # and return them as a list of BenchmarkArgs
    full_mjcf_list = ["xml/franka_emika_panda/panda.xml", "xml/unitree_g1/g1.xml", "xml/unitree_go2/go2.xml"]
    rasterizer_list = [True, False]
    full_batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    full_resolution_list = [(128, 128), (256, 256)]

    mjcf_list = full_mjcf_list
    resolution_list = full_resolution_list
    batch_size_list = full_batch_size_list
    # Batch data for resolution and batch size needs to be sorted in ascending order of resX x resY
    # so that if one resolution fails, all the resolutions, which are larger, will be skipped.
    resolution_list.sort(key=lambda x: x[0] * x[1])

    # Hardcoded parameters
    n_steps = 1
    camera_pos = (1.5, 0.5, 1.5)
    camera_lookat = (0.0, 0.0, 0.5)
    camera_fov = 45

    # Create a hierarchical dictionary to store all combinations
    batch_args_dict = {}

    # Build hierarchical structure
    for rasterizer in rasterizer_list:
        batch_args_dict[rasterizer] = {}
        for mjcf in mjcf_list:
            batch_args_dict[rasterizer][mjcf] = {}
            for batch_size in batch_size_list:
                batch_args_dict[rasterizer][mjcf][batch_size] = {}
                for resolution in resolution_list:
                    resX, resY = resolution
                    # Create benchmark args for this combination
                    args = BenchmarkArgs(
                        rasterizer=rasterizer,
                        n_envs=batch_size,
                        n_steps=n_steps,
                        resX=resX,
                        resY=resY,
                        camera_posX=camera_pos[0],
                        camera_posY=camera_pos[1],
                        camera_posZ=camera_pos[2],
                        camera_lookatX=camera_lookat[0],
                        camera_lookatY=camera_lookat[1],
                        camera_lookatZ=camera_lookat[2],
                        camera_fov=camera_fov,
                        mjcf=mjcf,
                        benchmark_result_file_path=benchmark_result_file_path
                    )
                    batch_args_dict[rasterizer][mjcf][batch_size][(resX,resY)] = args

    return batch_args_dict

def create_benchmark_result_file(continue_from_file_path):
    if continue_from_file_path is not None:
        if not os.path.exists(continue_from_file_path):
            raise FileNotFoundError(f"Continue from file not found: {continue_from_file_path}")
        print(f"Continuing from file: {continue_from_file_path}")
        return continue_from_file_path
    else:
        # Create benchmark result data file with header
        benchmark_data_directory = "logs/benchmark"
        if not os.path.exists(benchmark_data_directory):
            os.makedirs(benchmark_data_directory)
        benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_result_file_path = f"{benchmark_data_directory}/batch_benchmark_{benchmark_timestamp}.csv"
        with open(benchmark_result_file_path, "w") as f:
            f.write("result,mjcf,rasterizer,n_envs,n_steps,resX,resY,camera_posX,camera_posY,camera_posZ,camera_lookatX,camera_lookatY,camera_lookatZ,camera_fov,time_taken,time_taken_per_env,fps,fps_per_env\n")
        print(f"Created new benchmark result file: {benchmark_result_file_path}")
        return benchmark_result_file_path

def get_previous_runs(continue_from_file_path):
    if continue_from_file_path is None:
        return []
    
    # Read the existing benchmark data file
    df = pd.read_csv(continue_from_file_path)
    
    # Create a list of tuples containing run info and status
    previous_runs = []
    
    for _, row in df.iterrows():
        run_info = (
            row['mjcf'],
            row['rasterizer'],
            row['n_envs'],
            (row['resX'], row['resY']),
            row['result']  # 'succeeded' or 'failed'
        )
        previous_runs.append(run_info)
    
    return previous_runs

def run_batch_benchmark(batch_args_dict, benchmark_script_path, previous_runs=None):
    if previous_runs is None:
        previous_runs = []
        
    for rasterizer in batch_args_dict:
        for mjcf in batch_args_dict[rasterizer]:
            for batch_size in batch_args_dict[rasterizer][mjcf]:
                last_resolution_failed = False
                for resolution in batch_args_dict[rasterizer][mjcf][batch_size]:
                    if last_resolution_failed:
                        break
                    
                    # Check if this run was in a previous execution
                    run_info = (mjcf, rasterizer, batch_size, resolution)
                    skip_this_run = False
                    
                    for prev_run in previous_runs:
                        if run_info == prev_run[:4]:  # Compare only the run parameters, not the status
                            skip_this_run = True
                            if prev_run[4] == 'failed':
                                # Skip this and subsequent resolutions if it failed before
                                last_resolution_failed = True
                            break
                    
                    if skip_this_run:
                        continue
                        
                    # Run the benchmark
                    batch_args = batch_args_dict[rasterizer][mjcf][batch_size][resolution]
                    if batch_args.mjcf in ["xml/unitree_g1/g1.xml", "xml/unitree_go2/go2.xml"]:
                        if batch_args.resX == 256:
                            if batch_args.n_envs >= 512:
                                continue
                        elif batch_args.resX == 128:
                            if batch_args.n_envs >= 1024:
                                continue
                    else:
                        if batch_args.resX == 256 and batch_args.n_envs >= 1024:
                            continue
                    
                    # launch a process to run the benchmark
                    cmd = ["python3", benchmark_script_path]
                    if batch_args.rasterizer:
                        cmd.append("--rasterizer")
                    cmd.extend([
                        "--n_envs", str(batch_args.n_envs),
                        "--n_steps", str(batch_args.n_steps),
                        "--resX", str(batch_args.resX), 
                        "--resY", str(batch_args.resY),
                        "--camera_posX", str(batch_args.camera_posX),
                        "--camera_posY", str(batch_args.camera_posY),
                        "--camera_posZ", str(batch_args.camera_posZ), 
                        "--camera_lookatX", str(batch_args.camera_lookatX),
                        "--camera_lookatY", str(batch_args.camera_lookatY),
                        "--camera_lookatZ", str(batch_args.camera_lookatZ),
                        "--camera_fov", str(batch_args.camera_fov),
                        "--mjcf", batch_args.mjcf,
                        "--benchmark_result_file_path", batch_args.benchmark_result_file_path,
                        "--max_bounce", str(batch_args.max_bounce),
                        "--spp", str(batch_args.spp),
                    ])
                    print(" ".join(cmd))
                    try:
                        process = subprocess.Popen(cmd)
                        return_code = process.wait()
                        if return_code != 0:
                            raise subprocess.CalledProcessError(return_code, cmd)
                    except Exception as e:
                        print(f"Error running benchmark: {str(e)}")
                        last_resolution_failed = True
                        # Write failed result without timing data
                        with open(batch_args.benchmark_result_file_path, 'a') as f:
                            f.write(f'failed,{batch_args.mjcf},{batch_args.rasterizer},{batch_args.n_envs},{batch_args.n_steps},{batch_args.resX},{batch_args.resY},{batch_args.camera_posX},{batch_args.camera_posY},{batch_args.camera_posZ},{batch_args.camera_lookatX},{batch_args.camera_lookatY},{batch_args.camera_lookatZ},{batch_args.camera_fov},,,,\n')
                        break

def main():
    batch_benchmark_args = parse_args()
    benchmark_result_file_path = create_benchmark_result_file(batch_benchmark_args.continue_from)
    
    # Get list of previous runs if continuing from a previous run
    previous_runs = get_previous_runs(batch_benchmark_args.continue_from)

    # Run benchmark in batch
    current_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script_path = f"{current_dir}/{batch_benchmark_args.script}"
    if not os.path.exists(benchmark_script_path):
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script_path}")
        
    batch_args_dict = create_batch_args(benchmark_result_file_path)
    run_batch_benchmark(batch_args_dict, benchmark_script_path, previous_runs)

    # Generate plots
    plot_batch_benchmark(benchmark_result_file_path)

if __name__ == "__main__":
    main()
