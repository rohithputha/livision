
import pytest
import torch
from torch import nn

from runtime.inference.model_runner import ModelRunner


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_runner_execution():
    model = DummyModel()
    runner = ModelRunner(model, device="cpu")

    runner.load()

    batch = torch.randn(3, 4)
    output = runner.infer(batch)

    assert output.shape == (3, 2)
    assert output.requires_grad is False


def test_runner_sets_eval_mode():
    model = DummyModel()
    model.train()

    runner = ModelRunner(model, device="cpu")
    runner.load()

    assert model.training is False


def test_infer_before_load():
    runner = ModelRunner(DummyModel())

    with pytest.raises(RuntimeError):
        runner.infer(torch.randn(1, 4))



@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available"
)
def test_runner_on_mps():
    model = DummyModel()

    runner = ModelRunner(model, device="mps")
    runner.load()

    # The caller explicitly moves data to the GPU.
    batch = torch.randn(3, 4).to("mps")

    output = runner.infer(batch)

    assert output.shape == (3, 2)
    assert output.device.type == "mps"
    assert output.requires_grad is False
    assert torch.isfinite(output).all().item()
