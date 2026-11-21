
import numpy as np
from transforms.letterbox import letterbox
from runtime.core.types import Frame

def test_letterbox():
    image = np.full((120, 240, 3), 50, dtype=np.uint8)
    frame = Frame(image)
    result, meta = letterbox(frame, (224, 224))

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8

    assert meta.resized_hw == (112, 224)
    assert meta.padding == (56, 56, 0, 0)

    assert np.all(result[:56] == 114)
    assert np.all(result[56:168] == 50)
    assert np.all(result[168:] == 114)
