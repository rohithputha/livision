

import numpy as np
import torch
from runtime.core.types import Frame

image = np.arange(18, dtype = np.uint8).reshape(2,3,3)

frame = Frame(image)


nhwc = frame.to_tensor(layout="NHWC", normalize=False)
nchw = frame.to_tensor(layout="NCHW", normalize=False)

print ("NHWC: ")
print("Shape: ", nhwc.shape)
print("Strides: ", nhwc.stride())
print("Contiguous: ", nhwc.is_contiguous())

print("\nNCHW:")
print("Shape:", nchw.shape)
print("Strides:", nchw.stride())
print("Contiguous:", nchw.is_contiguous())


assert torch.equal(
    nhwc[0, 1, 2, :],
    nchw[0, :, 1, 2]
)


nchw_contiguous = nchw.contiguous()


print("\nContiguous NCHW:")
print("Strides:", nchw_contiguous.stride())
print("Contiguous:", nchw_contiguous.is_contiguous())