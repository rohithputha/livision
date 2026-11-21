
import numpy as np
import torch

from transforms.preprocess import (
    PreprocessConfig,
    preprocess_frame
)

from runtime.core.types import Frame
# Import Frame from your existing module.

def test_preprocess_frame():
    image = np.full(
        (120, 240, 3), 50, dtype=np.uint8
    )
    frame = Frame(image)

    config = PreprocessConfig(
        target_hw=(224, 224),
        layout="NCHW",
        normalize=True,
        pad_value=114
    )

    tensor, meta = preprocess_frame(frame, config)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert meta.padding == (56, 56, 0, 0)

    # Verify a padded pixel.
    assert torch.allclose(
        tensor[0, :, 0, 0],
        torch.full((3,), 114 / 255.0)
    )

    # Verify a pixel in the original image region.
    assert torch.allclose(
        tensor[0, :, 100, 100],
        torch.full((3,), 50 / 255.0)
    )

