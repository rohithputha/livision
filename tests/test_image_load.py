import cv2
import numpy as np

from ingestion.image import load_image

from runtime.core.types import ColorSpace

def test_load_image(tmp_path):
    path = tmp_path / "test.png"
    pixels = np.zeros((4,6,3), dtype = np.uint8)
    pixels[:, :]= [0, 0, 255]

    assert cv2.imwrite(str(path), pixels)
    frame = load_image(path)

    assert frame.data.shape == (4,6,3)
    assert frame.metadata.width ==6

    assert frame.metadata.height == 4
    assert frame.metadata.color_space == ColorSpace.RGB

