import time
import psutil
import torch
from train import SelfPruningNet

def profile_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfPruningNet(input_dim=64).to(device)
    x = torch.randn(100, 64).to(device)
    
    # Warmup
    _ = model(x)
    
    start_time = time.time()
    for _ in range(100):
        _ = model(x)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 100 * 1000
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    
    print(f"--- Profiling Results ---")
    print(f"Device: {device}")
    print(f"Avg Latency (100 samples): {avg_latency:.2f} ms")
    print(f"Memory Usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    profile_inference()
