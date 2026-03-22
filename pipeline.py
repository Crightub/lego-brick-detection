import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import csv
from PIL import Image
from ultralytics import YOLO

from ViTS16_Stage2.model import ViTClassifier


class LegoPipeline:

    def __init__(self,
                 stage1_path: str,
                 stage2_path: str,
                 labels_map_path: str,
                 num_classes: int = 200,
                 device: str = 'cuda',
                 conf: float = 0.01,
                 iou: float = 0.3,
                 padding: float = 0.12,
                 classifier_conf: float = 0.0):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf = conf
        self.iou = iou
        self.padding = padding
        self.classifier_conf = classifier_conf

        # Load Stage 1 — YOLO binary detector
        print("Loading Stage 1 (YOLO)...")
        self.detector = YOLO(stage1_path)

        # Load Stage 2 — ViT classifier
        print("Loading Stage 2 (ViT)...")
        self.classifier = ViTClassifier(num_classes=num_classes)
        self.classifier.load_state_dict(
            torch.load(stage2_path, map_location=self.device)
        )
        self.classifier.to(self.device)
        self.classifier.eval()

        self.idx_to_part_id = self._load_class_id_map(labels_map_path)

        self.crop_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_class_id_map(self, labels_map_path: str) -> dict:
        """
        Replicates ImageFolder's alphabetical folder sort to produce
        the same idx -> part_id mapping used during Stage 2 training.
        labels_map.csv format: part_name, part_id (integer)
        """
        part_ids = []
        with open(labels_map_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                part_ids.append(int(row[1]))

        # ImageFolder sorts folder names as strings alphabetically
        # Folder names are str(part_id) e.g. "1", "10", "100"...
        folder_names = sorted([str(pid) for pid in part_ids])
        idx_to_part_id = {idx: int(name) for idx, name in enumerate(folder_names)}
        return idx_to_part_id

    def _extract_crops(self, image: Image.Image, boxes: np.ndarray) -> torch.Tensor:
        """
        Crops each detected box from the image with padding,
        resizes to 128x128 and stacks into a batch tensor.
        """
        img_w, img_h = image.size
        crops = []

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            bw = xmax - xmin
            bh = ymax - ymin

            # Apply padding
            xmin = max(0, xmin - bw * self.padding)
            ymin = max(0, ymin - bh * self.padding)
            xmax = min(img_w, xmax + bw * self.padding)
            ymax = min(img_h, ymax + bh * self.padding)

            crop = image.crop((xmin, ymin, xmax, ymax))
            crop_tensor = self.crop_transform(crop)
            crops.append(crop_tensor)

        return torch.stack(crops).to(self.device)  # (N, 3, 128, 128)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        """
        Runs the full two-stage pipeline on a single PIL image.
        Returns dict with keys: boxes, labels, scores
        matching the format expected by your Evaluator.
        """
        yolo_results = self.detector.predict(
            source=np.array(image),
            conf=self.conf,
            iou=self.iou,
            max_det=500,
            verbose=False
        )[0]

        boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()  # (N, 4)
        yolo_scores = yolo_results.boxes.conf.cpu().numpy()  # (N,)

        if len(boxes_xyxy) == 0:
            return {
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros(0, dtype=torch.long),
                'scores': torch.zeros(0),
            }

        # Stage 2 — classify each crop
        crops = self._extract_crops(image, boxes_xyxy)
        _, logits = self.classifier(crops)
        probs = F.softmax(logits, dim=1)

        classifier_scores, folder_indices = probs.max(dim=1)
        classifier_scores = classifier_scores.cpu().numpy()
        folder_indices = folder_indices.cpu().numpy()

        # Map ImageFolder indices back to real part IDs
        part_ids = np.array([
            self.idx_to_part_id[idx] for idx in folder_indices
        ])

        # Combined confidence score
        combined_scores = yolo_scores * classifier_scores

        # Apply classifier confidence threshold if set
        if self.classifier_conf > 0.0:
            mask = classifier_scores >= self.classifier_conf
            boxes_xyxy = boxes_xyxy[mask]
            part_ids = part_ids[mask]
            combined_scores = combined_scores[mask]

        return {
            'boxes': torch.tensor(boxes_xyxy, dtype=torch.float32),
            'labels': torch.tensor(part_ids, dtype=torch.long),
            'scores': torch.tensor(combined_scores, dtype=torch.float32),
        }
