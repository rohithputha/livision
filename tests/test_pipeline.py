import cv2

import numpy as np

import torch


from ingestion.image import load_image
from runtime.core.types import Frame, InferenceRequest, InferenceResult



def test_image_to_tensor_pipeline(tmp_path):
    path = tmp_path / "test.png"
    image = np.zeros((4,6,3), dtype = np.uint8)
    image[:, :] = [0, 0, 255]

    cv2.imwrite(str(path), image)
    frame = load_image(path)

    tensor = frame.to_tensor()
    assert tensor.shape == (1,3, 4, 6)

    assert tensor.dtype == torch.float32

    torch.testing.assert_close(
        tensor[0, :, 0, 0],
        torch.tensor([1.0, 0.0, 0.0])
    )

    request = InferenceRequest(
        req_id = "req1",
        model_id = "detector",
        frame = frame
    )

    assert request.frame is frame

    result = InferenceResult(
        req_id=request.req_id,
        model_id=request.model_id,
        outputs={}
    )

    assert result.req_id == request.req_id


def test_nhwc_without_normalization(tmp_path):
    path = tmp_path / "test.png"

    image = np.zeros((2, 3, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)

    frame = load_image(path)

    tensor = frame.to_tensor(
        layout="NHWC",
        normalize=False
    )

    assert tensor.shape == (1, 2, 3, 3)
    assert tensor.dtype == torch.uint8


