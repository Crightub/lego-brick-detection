import torch
from torchvision.transforms import v2


def get_transforms(is_train: bool):
    if is_train:
        return v2.Compose([
            v2.ToImage(),
            v2.RandomPhotometricDistort(p=1),
            v2.RandomZoomOut(),
            v2.RandomIoUCrop(),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float, scale=True)
        ])

    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float, scale=True)
    ])
