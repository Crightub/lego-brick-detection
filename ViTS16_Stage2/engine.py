import time
import torch
import numpy as np
from torch.cuda.amp import autocast
from collections import defaultdict


def train_single_epoch(model, optimizer, arcface_loss, dataloader, device, epoch, scaler):
    model.train()
    arcface_loss.train()
    print(f'Epoch: {epoch}')

    epoch_start = time.time()
    use_amp = device.type == 'cuda'
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            embeddings, logits = model(images)
            arc_loss = arcface_loss(embeddings, labels)
            ce_loss = torch.nn.functional.cross_entropy(logits, labels)

            # Combined loss — ArcFace shapes the embedding space,
            # CE ensures the classifier head trains correctly
            loss = arc_loss + ce_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch [{batch_idx}/{len(dataloader)}]"
                  f"  loss: {loss.item():.4f}"
                  f"  arc: {arc_loss.item():.4f}"
                  f"  ce: {ce_loss.item():.4f}")

    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch}] "
          f"loss: {running_loss / len(dataloader):.4f}  "
          f"acc: {correct / total:.4f}  "
          f"time: {epoch_time / 60:.1f} min")


@torch.inference_mode()
def evaluate(model, dataloader, device, class_id_map: dict):
    """
    Computes top-1 and top-5 accuracy plus per-class breakdown.
    class_id_map: {imagefolder_idx -> real part ID}
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Per-class tracking using real part IDs
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        _, logits = model(images)

        # Top-1
        top1_preds = logits.argmax(dim=1)
        correct_top1 += (top1_preds == labels).sum().item()

        # Top-5
        top5_preds = logits.topk(5, dim=1).indices
        for i, label in enumerate(labels):
            if label in top5_preds[i]:
                correct_top5 += 1

        # Per-class breakdown with real part IDs
        for pred, label in zip(top1_preds.cpu(), labels.cpu()):
            real_id = class_id_map[label.item()]
            class_correct[real_id] += int(pred == label)
            class_total[real_id] += 1

        total += labels.size(0)

    top1 = correct_top1 / total
    top5 = correct_top5 / total

    print(f"\n{'=' * 40}")
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")

    # Print worst performing classes — most useful for spotting mirror confusion
    print(f"\nWorst 10 classes:")
    print(f"{'Part ID':<12} | {'Acc':<8} | {'Correct'}/{'{Total}'}")
    print("-" * 40)
    per_class_acc = {
        cid: class_correct[cid] / class_total[cid]
        for cid in class_total
    }
    for cid, acc in sorted(per_class_acc.items(), key=lambda x: x[1])[:10]:
        print(f"  ID {cid:<8} | {acc:.4f}  | {class_correct[cid]}/{class_total[cid]}")

    print("=" * 40)
    return top1