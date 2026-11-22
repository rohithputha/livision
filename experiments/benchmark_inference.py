

import time
import numpy as np

import torch

from torchvision.models import resnet18, ResNet18_Weights
from ingestion.image import load_image

def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()

def benchmark_model(model, batch, device, warmup = 10, iterations = 50):
    durations = []

    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)

        sync(device)

    for _ in range(iterations):
        sync(device)
        start = time.perf_counter_ns()

        output = model(batch)
        sync(device)
        end = time.perf_counter_ns()


        durations.append((end- start)/ 1e6)

    assert output.shape == (1, 1000)

    return (
        np.median(durations),
        np.percentile(durations, 95),
    )

def run_on_device(device, cpu_batch, weights):
    print(f"\nBenchmarking: {device}")

    model = resnet18(weights=weights).to(device)
    model.eval()

    batch = cpu_batch.to(device)

    p50, p95 = benchmark_model(model, batch, device)

    print(f"p50: {p50:.3f} ms")
    print(f"p95: {p95:.3f} ms")



def main():
    weights = ResNet18_Weights.DEFAULT

    # Use the same input with the official transforms
    frame = load_image("data/raw/sample.jpg")
    cpu_batch = weights.transforms()(
        frame.to_tensor(layout="NCHW", normalize=False)
    )

    run_on_device(torch.device("cpu"), cpu_batch, weights)

    if torch.backends.mps.is_available():
        run_on_device(torch.device("mps"), cpu_batch, weights)
    else:
        print("MPS is unavailable on this setup")


if __name__ == "__main__":
    main()



