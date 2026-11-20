

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ColorSpace(str, Enum):
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"


@dataclass
class FrameMetadata:
    width: int
    height: int
    channels: int
    color_space : ColorSpace

    source_id: str | None = None
    frame_index: int | None = None

    timestamp_ns: int | None = None

@dataclass
class Frame: 
    data: np.ndarray

    metadata: FrameMetadata


    def __post_init__(self):
        expected = (
            self.metadata.height,
            self.metadata.width,
            self.metadata.channels
        ) 

        if self.data.shape != expected:
            raise ValueError(
                f"Expected shape {expected}, "
                f"Received {self.data.shape}"
            )

        if self.data.dtype != np.uint8:
            raise TypeError(f"Expected unit8, got {self.data.dtype}")

        expected_channels = {
            ColorSpace.RGB: 3,
            ColorSpace.BGR: 3,
            ColorSpace.GRAY: 1,
        }

        if self.metadata.channels!= expected_channels[self.metadata.color_space]:
            raise ValueError("Invalid Channel Count")

        