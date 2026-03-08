import time
import torch
from torch.cuda.amp import autocast, GradScaler

ACCUMULATION_STEPS = 4


def train_single_epoch(model, optimizer, dataloader_train, device, epoch, scaler):
    model.train()
    print(f'Epoch: {epoch}')
    optimizer.zero_grad()

    epoch_start = time.time()
    use_amp = device.type == 'cuda'  # Only use mixed precision on CUDA

    for batch_idx, (images, targets) in enumerate(dataloader_train):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) / ACCUMULATION_STEPS

        print(f"Epoch [{epoch}] Batch [{batch_idx}]")
        for k, v in loss_dict.items():
            print(f"  - {k}: {v.item():.4f}")

        scaler.scale(losses).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Handle remaining batches
        if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    epoch_time = time.time() - epoch_start
    print(f"Epoch time: {epoch_time / 60:.1f} min")
