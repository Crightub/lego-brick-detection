import torch

def train_single_epoch(model, optimizer, dataloader_train, device, epoch):
    model.train()

    print(f'Epoch: {epoch}')

    for batch_idx, (images, targets) in enumerate(dataloader_train):
        images = [image.to(device) for image in images]
        targets = [{k : v.to(device) if isinstance(v, torch.Tensor) else v for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        print(f"Epoch [{epoch}] Batch [{batch_idx}]")
        for k, v in loss_dict.items():
            print(f"  - {k}: {v.item():.4f}")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
