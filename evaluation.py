import torch
import os
import time
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from pipeline import LegoPipeline
from ViTS16_Stage2.dataset import get_transforms
from FastRCNN.eval import Evaluator
from b200_lego_dataset import LegoImageDataset


def evaluate_pipeline(pipeline: LegoPipeline,
                      dataset: LegoImageDataset,
                      device: str = 'cuda'):

    evaluator = Evaluator()
    total_time = 0.0

    print(f"Running pipeline evaluation on {len(dataset)} images...")

    for idx in range(len(dataset)):
        # Get ground truth
        image_tensor, target = dataset[idx]

        # Convert to PIL for pipeline input
        image = to_pil_image(image_tensor)

        # Run two-stage pipeline
        t0 = time.time()
        pred = pipeline.predict(image)
        total_time += time.time() - t0

        target_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in target.items()
        }

        res = {target_cpu['image_id'].item(): (target_cpu, pred)}
        evaluator.update(res)

        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(dataset)} images")

    print(f"\nTotal inference time: {total_time:.1f}s  "
          f"({total_time / len(dataset) * 1000:.1f}ms per image)")

    evaluator.accumulate()
    evaluator.summary()


if __name__ == '__main__':
    pipeline = LegoPipeline(
        stage1_path='model/stage1/weights/best.pt',
        stage2_path='model/stage2/best_model.pth',
        labels_map_path='data/labels_map.csv',
        device='cpu',
        conf=0.01,
        iou=0.3,
    )

    dataset_val = LegoImageDataset(
        annotation_dir='data/val/annotations',
        image_dir='data/val/images',
        labels_path='data/labels_map.csv',
        transforms=None
    )

    evaluate_pipeline(pipeline, dataset_val)