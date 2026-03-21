import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import ArcFaceLoss

from ViTS16_Stage2.model import ViTClassifier
from ViTS16_Stage2.dataset import load_train_set, load_val_set, get_class_id_map
from ViTS16_Stage2.engine import train_single_epoch, evaluate


def main(args):
    device = torch.device(args.device)
    print('Using device:', device)

    print('Loading data...')
    dataset_train = load_train_set(args.crops_train_dir)
    dataset_val   = load_val_set(args.crops_val_dir)

    num_classes = len(dataset_train.classes)
    class_id_map = get_class_id_map(dataset_train)
    print(f'Number of classes: {num_classes}')

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print('Loading model...')
    model = ViTClassifier(num_classes=num_classes).to(device)

    # ArcFace loss — embedding_dim must match ViT-S output (384)
    arcface_loss = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=384,
        margin=28.6,   # default, tune later if needed
        scale=64       # default
    ).to(device)

    print('Setup training...')
    # Combine model params and ArcFace params in one optimizer
    # ArcFace has its own learnable weight matrix that must be trained
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': arcface_loss.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        arcface_loss.load_state_dict(checkpoint['arcface_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1

    best_top1 = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_single_epoch(
            model, optimizer, arcface_loss,
            dataloader_train, device, epoch, scaler
        )
        lr_scheduler.step()

        top1 = evaluate(model, dataloader_val, device, class_id_map)
        save_checkpoint(model, optimizer, lr_scheduler,
                        scaler, arcface_loss, epoch, args.model_out_dir)

        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(),
                       os.path.join(args.model_out_dir, 'best_model.pth'))
            print(f"New best top-1: {best_top1:.4f} — saved best_model.pth")


def save_checkpoint(model, optimizer, scheduler, scaler, arcface_loss, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'arcface_state_dict': arcface_loss.state_dict(),
    }
    if epoch % 5 == 0:
        torch.save(checkpoint, os.path.join(out_dir, f'model_epoch_{epoch}.pth'))
    torch.save(checkpoint, os.path.join(out_dir, 'latest_model.pth'))
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(resume_path):
    if not os.path.exists(resume_path):
        print("Checkpoint not found, training from scratch.")
        return None
    return torch.load(resume_path)


local_train_preset = argparse.Namespace(
    crops_train_dir='../data/train/crops',
    crops_val_dir='../data/val/crops',
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=64,
    epochs=30,
    milestones=[15, 25],
    gamma=0.1,
    device='cpu',
    model_out_dir='../model/stage2',
    resume=False,
    start_epoch=0,
)

if __name__ == '__main__':
    main(local_train_preset)