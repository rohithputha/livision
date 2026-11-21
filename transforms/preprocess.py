


from dataclasses import dataclass
import torch
from typing import Literal
from runtime.core.types import Frame
from transforms.letterbox import LetterboxMetadata, letterbox

@dataclass(frozen=True)
class PreprocessConfig:
    target_hw: tuple[int, int]
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    normalize: bool = True
    pad_value: int = 114

def preprocess_frame(
        frame : Frame,
        config: PreprocessConfig 
) -> tuple[torch.Tensor, LetterboxMetadata]:

    
    image, lmetadata = letterbox(frame, config.target_hw, config.pad_value)
    frame = Frame(image)
    frame_tensor = frame.to_tensor(config.layout, config.normalize)

    return (frame_tensor, lmetadata)
