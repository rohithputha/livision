import sys
import torch

from torchvision.models import resnet18, ResNet18_Weights

from ingestion.image import load_image
from transforms.preprocess import PreprocessConfig, preprocess_frame


def main(image_path: str):
    frame = load_image(image_path)
    print("Original shape: ", frame.data.shape)

    config = PreprocessConfig(
        target_hw=(224, 224),
        layout="NCHW",
        normalize=True
    )

    tensor, metadata = preprocess_frame(frame, config)

    print("Tensor shape: ", tensor.shape)
    print("Letterbox metadata: ", metadata)

    mean = torch.tensor([
        0.485, 0.456, 0.406
    ]).view(1,3, 1,1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    tensor = (tensor - mean) / std


    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights)

    model.eval()

    with torch.inference_mode():
        logits = model (tensor)
        prob = torch.softmax(logits, dim =1)

    scores, indices = torch.topk(
        prob, k = 5, dim = 1
    )

    categories  = weights.meta["categories"]
    for score, index in zip(scores[0], indices[0]):
        print(
            categories[index.item()],
            f"{score.item() * 100:.2f}%"
        )

    raw_tensor = frame.to_tensor(
    layout="NCHW",
    normalize=False
    )

    # Apply the exact transforms specified by the weights
    official_tensor = weights.transforms()(raw_tensor)

    print("Official shape:", official_tensor.shape)

    with torch.inference_mode():
        official_logits = model(official_tensor)

    official_probs = torch.softmax(official_logits, dim=1)

    scores, indices = official_probs.topk(5, dim=1)

    print("\nOfficial preprocessing predictions:")

    for score, index in zip(scores[0], indices[0]):
        print(
            categories[index.item()],
            f"{score.item() * 100:.2f}%"
        )


if __name__ == "__main__":
    main(sys.argv[1])
