import time
from collections import defaultdict
from typing import Dict, Tuple

import torch
import numpy as np


class Evaluator:

    def __init__(self):
        self.images_ids = []
        self.label_to_pred = defaultdict(list)
        self.label_to_gt = defaultdict(lambda: defaultdict(list))
        self.label_ap = defaultdict(list)
        self.eval = {}

    def update(self, res: Dict[str, Tuple[Dict, Dict]]):
        # data is already on cpu
        image_ids = list(np.unique(res.keys()))
        self.images_ids.extend(image_ids)

        for image_id, (target, pred) in res.items():

            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            target_boxes = target['boxes'].cpu().numpy()
            target_labels = target['labels'].cpu().numpy()

            for i, label in enumerate(pred_labels):
                self.label_to_pred[label].append({
                    'image_id': image_id,
                    'boxes': pred_boxes[i],
                    'score': pred_scores[i],
                })

            for i, label in enumerate(target_labels):
                self.label_to_gt[label.item()][image_id].append(target_boxes[i])

    def accumulate(self):
        all_ap = []

        for label, gt_by_image in self.label_to_gt.items():
            total_gt = sum(len(boxes) for boxes in gt_by_image.values())
            if total_gt == 0:
                continue

            preds = self.label_to_pred.get(label, [])
            preds.sort(key=lambda x: x['score'], reverse=True)

            visited_gt = {image_id: [False] * len(boxes) for image_id, boxes in gt_by_image.items()}
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for i, pred in enumerate(preds):
                image_id = pred['image_id']
                boxes = pred['boxes']

                gts = gt_by_image.get(image_id, [])

                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gts):
                    iou = self._iou(boxes, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou > 0.2:
                    if not visited_gt[image_id][best_gt_idx]:
                        visited_gt[image_id][best_gt_idx] = True
                        tp[i] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            tp_sum = np.cumsum(tp)
            fp_sum = np.cumsum(fp)

            recall = tp / total_gt
            precision = tp / (tp_sum + fp_sum + 1e-10)

            ap = self._calculate_ap(recall, precision)
            all_ap.append(ap)
            self.label_ap[label] = ap

        self.mAP = np.mean(all_ap) if all_ap else 0

    def _calculate_ap(self, recall, precision):
        """
            Computes the Average Precision (AP) using the all-points interpolation method.
        """
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = np.maximum(mpre[i], mpre[i + 1])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def summary(self, labels_map=None):
        """
        Prints a formatted report of the detection performance.
        labels_map: Optional dict mapping {id: "name"} to show brick names.
        """
        print("\n" + "=" * 40)
        print(f"{'Lego Part Performance (IoU=0.5)':^40}")
        print("=" * 40)
        print(f"{'Label/Name':<25} | {'AP':<10}")
        print("-" * 40)

        sorted_results = sorted(self.label_ap.items(), key=lambda x: x[1], reverse=True)

        for label, ap in sorted_results:
            display_name = labels_map.get(label, f"ID {label}") if labels_map else f"ID {label}"
            print(f"{display_name:<25} | {ap:>8.2%}")

        print("-" * 40)
        print(f"{'MEAN AVERAGE PRECISION (mAP)':<25} | {self.mAP:>8.2%}")
        print("=" * 40 + "\n")

    def _iou(self, pred_boxes, target_boxes) -> float:
        """
            Computes the intersection over union (IoU) of self and target.
            Boxes: x_min, y_min, x_max, y_max
        """
        # Compute area of intersection
        intersection_width = min(pred_boxes[2], target_boxes[2]) - max(pred_boxes[0], target_boxes[0])
        intersection_height = min(pred_boxes[3], target_boxes[3]) - max(pred_boxes[1], target_boxes[1])

        if intersection_width <= 0 or intersection_height <= 0:
            return 0.0

        intersection_area = intersection_width * intersection_height
        pred_area = (pred_boxes[2] - pred_boxes[0]) * (pred_boxes[3] - pred_boxes[1])
        target_area = (target_boxes[2] - target_boxes[0]) * (target_boxes[3] - target_boxes[1])
        union_area = pred_area + target_area - intersection_area
        iou = intersection_area / union_area
        return iou


@torch.inference_mode()
def evaluate(model, dataloader):
    cpu_device = torch.device("cpu")
    model.eval()

    total_evaluator_time = 0.0
    evaluator = Evaluator()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [image.to(cpu_device) for image in images]
        targets = [{k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        preds = model(images, targets)
        preds = [{k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in preds]

        res = {target['image_id']: (target, pred) for target, pred in zip(targets, preds)}

        evaluator_time = time.time()
        evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        total_evaluator_time += evaluator_time

    print(f'Evaluator Time: {total_evaluator_time}')
    evaluator.accumulate()
    evaluator.summary()
