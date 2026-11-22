

import torch
from torch import nn

class ModelRunner:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self._loaded = False

    def load(self):
        model = self.model.to(self.device)
        model.eval()
        self._loaded = True

    def infer(self, batch: torch.Tensor):
        if not self._loaded:
            raise RuntimeError("Model not loaded into the device yet")

        if self.device.type != batch.device.type:
            print(self.device)
            print(batch.device)
            raise RuntimeError("Batch device does not match model device")

        with torch.inference_mode():
            output = self.model(batch)

        return output