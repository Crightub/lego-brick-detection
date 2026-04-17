from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_transforms(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

def load_train_set(crops_train_dir: str):
    return ImageFolder(
        root=crops_train_dir,
        transform=get_transforms(is_train=True)
    )


def load_val_set(crops_val_dir: str):
    return ImageFolder(
        root=crops_val_dir,
        transform=get_transforms(is_train=False)
    )


def get_class_id_map(dataset: ImageFolder) -> dict:
    return {v: int(k) for k, v in dataset.class_to_idx.items()}