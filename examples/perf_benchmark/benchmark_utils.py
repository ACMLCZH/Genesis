import torch

def fill_gpu_cache_with_random_data():
    # 100 MB of random data
    dummy_data = torch.rand(100, 1024, 1024, device="cuda")
    # Make some random data manipulation to the entire tensor
    dummy_data = dummy_data.sqrt()