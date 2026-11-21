import time
import numpy as np

import torch

from runtime.core.types import Frame
from runtime.core.batching import batch_tensors
from transforms.letterbox import letterbox

from transforms.preprocess import PreprocessConfig, preprocess_frame


def benchmark(fn, iterations = 50, warmup= 10):
    for _ in range(warmup):
        fn()

    durations = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        durations.append((end-start)/1e6)

    return (
        np.median(durations),
        np.percentile(durations, 95)
    )


rng = np.random.default_rng(42)

frames = [
    Frame(rng.integers(
        0, 256, (480, 640,3), dtype = np.uint8
    ))
    for _ in range(10)
]


config = PreprocessConfig(
    target_hw=(224, 224),
    layout="NCHW",
    normalize=True
)

raw_config = PreprocessConfig(
    target_hw=(224, 224),
    layout="NCHW",
    normalize=False
)

resized = [
    Frame(letterbox(f, (224, 224))[0])
    for f in frames
]


tensors = [
    f.to_tensor(layout = "NCHW", normalize = True)
    for f in frames
]

def pipeA():
    results = [
        preprocess_frame(f, config)[0]
        for f in frames
    ]
    return batch_tensors(results)

def pipeB():
    results = [
        preprocess_frame(f, raw_config)[0]
        for f in frames
    ]

    batch = batch_tensors(results)
    return batch.float() / 255.0



def report(name, fn):
    p50, p95 = benchmark(fn)
    print(f"{name:25s} p50={p50:8.3f} ms"
          f"  p95={p95:8.3f} ms")



if __name__ == "__main__":
    # Verify the alternatives produce the same values
    assert torch.allclose(pipeA(), pipeB())

    report("Letterbox (1 frame)",
           lambda: letterbox(frames[0], (224, 224)))

    report("Tensor conversion (1)",
           lambda: resized[0].to_tensor(
               layout="NCHW", normalize=True
           ))

    report("Batching (4 tensors)",
           lambda: batch_tensors(tensors))

    report("Pipeline A (4 frames)", pipeA)
    report("Pipeline B (4 frames)", pipeB)


    uint8_tensors = [
    f.to_tensor(layout="NCHW", normalize=False)
    for f in resized
    ]

    resized_frames = [
    Frame(letterbox(f, (224, 224))[0])
    for f in frames
    ]

    uint8_tensors = [
        f.to_tensor(layout="NCHW", normalize=False)
        for f in resized_frames
    ]

    float_tensors = [
        t.float() / 255.0
        for t in uint8_tensors
    ]

    # Verify that the comparison is valid
    for f32, u8 in zip(float_tensors, uint8_tensors):
        assert f32.shape == u8.shape == (1, 3, 224, 224)

    print("Float32:", float_tensors[0].shape)
    print("Uint8:", uint8_tensors[0].shape)

    report(
        "Concat float32",
        lambda: torch.cat(float_tensors, dim=0)
    )

    report(
        "Concat uint8",
        lambda: torch.cat(uint8_tensors, dim=0)
    )

    uint8_batch = torch.cat(uint8_tensors, dim=0)

    report(
        "Batch normalization",
        lambda: uint8_batch.float() / 255.0
    )
