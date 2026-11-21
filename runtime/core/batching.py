import torch


def batch_tensors(
        tensors: list[torch.Tensor],
)-> torch.Tensor:

    if tensors is None or len(tensors) == 0:
        raise ValueError("No tensors in the batch")

    
    for t in tensors:
        try:
            assert len(t.shape) == 4 and t.shape[0] == 1
            assert t.shape[1] == tensors[0].shape[1]
            assert t.shape[2] == tensors[0].shape[2]
            assert t.shape[3] == tensors[0].shape[3]
            assert t.dtype == tensors[0].dtype
            assert t.device == tensors[0].device
        except:
            raise ValueError("Tensors not equivalent")

    return torch.cat(tensors, dim = 0)


    