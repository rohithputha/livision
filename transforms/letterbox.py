
from dataclasses import dataclass
import cv2
import numpy as np
from runtime.core.types import Frame

@dataclass(frozen= True)
class LetterboxMetadata:
    scale: float
    resized_hw : tuple[int, int]
    padding: tuple[int, int, int, int]



def letterbox(
        frame: Frame,
        target_hw: tuple[int,int], 
        pad_value: int = 114
)-> tuple[np.ndarray, LetterboxMetadata]:
    image:np.ndarray = frame.get_frame()
    h, w = image.shape[:2]
    target_h, target_w = target_hw

    scale = min(target_h/ h, target_w/ w)

    rh = round(scale * h)
    rw = round(scale * w)

    resize = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - rw
    pad_h = target_h - rh

    top = pad_h//2
    bottom = pad_h - top

    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resize, top, bottom, left , right,
        borderType=cv2.BORDER_CONSTANT,
        value= (pad_value, pad_value, pad_value)
    )

    return (
        padded,
        LetterboxMetadata(
            scale,
            (rh, rw),
            (top, bottom, left, right),
        )
    )
