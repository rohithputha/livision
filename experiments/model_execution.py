


import torch

from torchvision.models import resnet18, ResNet18_Weights

def main():
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights = weights)

    model.eval()

    batch = torch.rand(1,3, 224, 224)

    print("Input shape", batch.shape)

    with torch.inference_mode():
        output = model(batch)

    print("output shape: ", output.shape)
    print("Output dtype", output.dtype)

    assert output.shape == (1,1000)

    assert torch.isfinite(output).all()

    print("inference successful")



if __name__ == "__main__":
    main()