

from dataclasses import dataclass
from enum import Enum

import numpy as np
from typing import Literal

import torch
from typing import Any

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


    def __init__(self, data, metadata= None):
        self.data = data

        if metadata is None:
            h, w, channels = data.shape
            self.metadata = FrameMetadata(
                        width=w,
                        height=h,
                        channels=channels,
                        color_space= ColorSpace.RGB,
                        source_id=None
                    )
        else:
            self.metadata = metadata


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


    def to_tensor(self, layout: Literal["NCHW", "NHWC"] = "NCHW", normalize: bool = True)-> torch.Tensor:

        if layout not in ("NCHW", "NHWC"):
            raise ValueError(f"Unsupported layout: {layout}")

        tensor = torch.from_numpy(self.data)

        if self.metadata.color_space == ColorSpace.BGR:
            tensor = tensor[...,[2, 1, 0]]

        if normalize:
            tensor = tensor.float()/255.0

        if layout == "NCHW":
            tensor = tensor.permute(2, 0, 1 )

        tensor = tensor.unsqueeze(0)
        return tensor

    def get_frame(self):
        return self.data
    


@dataclass(frozen = True)
class InferenceRequest:
    req_id: str
    model_id: str
    frame: Frame


@dataclass
class InferenceResult:
    req_id: str
    model_id: str
    outputs: dict[str, Any]
    latency_ms: float | None = None

