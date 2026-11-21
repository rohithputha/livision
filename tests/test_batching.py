
import pytest
import torch

from runtime.core.batching import batch_tensors


def test_batch_tensors():
    a = torch.zeros(1, 3, 224, 224)
    b = torch.ones(1, 3, 224, 224)

    result = batch_tensors([a, b])

    assert result.shape == (2, 3, 224, 224)
    assert torch.equal(result[0], a[0])
    assert torch.equal(result[1], b[0])


def test_empty_batch():
    with pytest.raises(ValueError):
        batch_tensors([])


def test_mismatched_shapes():
    a = torch.zeros(1, 3, 224, 224)
    b = torch.zeros(1, 3, 128, 128)

    with pytest.raises(ValueError):
        batch_tensors([a, b])


def test_mismatched_dtypes():
    a = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    b = torch.zeros(1, 3, 224, 224, dtype=torch.float16)

    with pytest.raises(ValueError):
        batch_tensors([a, b])


def test_missing_batch_dimension():
    a = torch.zeros(3, 224, 224)

    with pytest.raises(ValueError):
        batch_tensors([a])
