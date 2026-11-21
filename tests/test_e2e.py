
import numpy as np
import torch

from runtime.core.types import Frame
from runtime.core.batching import batch_tensors
from transforms.preprocess import (
    PreprocessConfig,
    preprocess_frame
)



from transforms.letterbox import letterbox


def test_multiple_frames_end_to_end():
    # Two images of different dimensions and colors
    image1 = np.full((120, 240, 3), 50, dtype=np.uint8)
    image2 = np.full((100, 100, 3), 200, dtype=np.uint8)

    frames = [Frame(image1), Frame(image2)]

    config = PreprocessConfig(
        target_hw=(224, 224),
        layout="NCHW",
        normalize=True
    )

    tensors = []
    metadata = []

    for frame in frames:
        tensor, meta = preprocess_frame(frame, config)
        tensors.append(tensor)
        metadata.append(meta)

    batch = batch_tensors(tensors)

    assert batch.shape == (2, 3, 224, 224)
    assert batch.dtype == torch.float32

    # Check that each frame retained its pixel values
    assert torch.allclose(
        batch[0, :, 100, 100],
        torch.full((3,), 50 / 255.0)
    )

    assert torch.allclose(
        batch[1, :, 100, 100],
        torch.full((3,), 200 / 255.0)
    )

    # The first image requires vertical padding
    assert metadata[0].padding == (56, 56, 0, 0)

    # The square image needs no padding
    assert metadata[1].padding == (0, 0, 0, 0)





def test_odd_padding():
    image = np.zeros((120, 240, 3), dtype=np.uint8)

    result, meta = letterbox(Frame(image), (225, 224))

    assert result.shape == (225, 224, 3)
    assert meta.padding == (56, 57, 0, 0)
