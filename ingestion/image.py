from pathlib import Path


import cv2

from runtime.core.types import (
    Frame,
    FrameMetadata,
    ColorSpace
)


def load_image(path: str | Path)-> Frame:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise ValueError(f"Cannot load image : {path}")

    h, w, channels = image.shape

    metadata = FrameMetadata(
        width=w,
        height=h,
        channels=channels,
        color_space= ColorSpace.RGB,
        source_id=str(path)
    )

    return Frame(
        data = image,
        metadata=metadata
    )

