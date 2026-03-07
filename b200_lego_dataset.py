import os
from typing import Optional

from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_image, ImageReadMode
import xml.etree.ElementTree as ET
import torch
import csv

from presets import get_transforms

class LegoImageDataset(Dataset):
    """
        Defines the dataset class for loading image data together with the annotations.
    """
    def __init__(self,
                 annotation_dir: str,
                 image_dir: str,
                 labels_path: str,
                 transforms=None,
                 max_size=None):
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.transforms = transforms
        self.filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.labels_map = self._generate_label_map(labels_path)

        if max_size is not None:
            self.filenames = self.filenames[:max_size]

    def label_count(self):
        # +1 for background class
        return len(self.labels_map) + 1

    def __len__(self):
        return len(self.filenames)

    def _generate_label_map(self, labels_path: str):
        labels_map = {}
        with open(labels_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                labels_map[row[0]] = int(row[1])

        return labels_map

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = os.path.join(self.image_dir, f"{filename}.png")
        image = decode_image(image_path, mode=ImageReadMode.RGB)

        # Load annotations and convert to pytorch standard
        annotation_path = os.path.join(self.annotation_dir, f"{filename}.xml")

        tree = ET.parse(annotation_path)

        labels = []
        boxes = []
        areas = []

        for obj in tree.findall("object"):
            name = obj.find("name").text
            # color = obj.find("color").text
            box = obj.find("bndbox")

            label = self.labels_map[name]

            coords = [
                float(box.find("xmin").text),
                float(box.find("ymin").text),
                float(box.find("xmax").text),
                float(box.find("ymax").text)
            ]

            # Compute area: (x_max - x_min) * (y_max - y_min)
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])

            labels.append(label)
            boxes.append(coords)
            areas.append(area)

        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(boxes, format="xyxy", canvas_size=(image.shape[1], image.shape[2]))
        target['labels'] = torch.tensor(labels)
        target['area'] = torch.tensor(areas)
        target['iscrowd'] = torch.zeros(len(labels), dtype=torch.long)
        target['image_id'] = torch.tensor([index])

        sample = image, target

        if self.transforms is not None:
            sample = self.transforms(*sample)

        return sample


def load_train_set(max_size : Optional[int] = None) -> LegoImageDataset:
    return LegoImageDataset(annotation_dir="data/train/annotations",
                            image_dir="data/train/images",
                            labels_path="data/labels_map.csv",
                            transforms=get_transforms(is_train=True),
                            max_size=max_size)


def load_test_set() -> LegoImageDataset:
    return LegoImageDataset(annotation_dir="data/test/annotations",
                            image_dir="data/test/images",
                            labels_path="data/labels_map.csv",
                            transforms=get_transforms(is_train=False))


def load_val_set(max_size: Optional[int] = None) -> LegoImageDataset:
    return LegoImageDataset(annotation_dir="data/val/annotations",
                            image_dir="data/val/images",
                            labels_path="data/labels_map.csv",
                            transforms=get_transforms(is_train=False),
                            max_size=max_size)
