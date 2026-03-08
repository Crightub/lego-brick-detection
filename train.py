import argparse
import os
import time

import torch
import torchvision
from torch import GradScaler
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from b200_lego_dataset import load_train_set, load_val_set
from engine import train_single_epoch
from eval import evaluate


def main(args):
    device = torch.device(args.device)
    print('Using device:', device)

    print('Loading data...')
    dataset_train = load_train_set(args.train_size)
    dataset_val = load_val_set(args.val_size)

    num_classes = dataset_train.label_count()
    print(f'Number of classes: {num_classes}')

    print('Create DataLoader...')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=lambda x: tuple(zip(*x)))
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    print('Loading Model...')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights,
                                                                 rpn_post_nms_top_n_train=2000,
                                                                 rpn_post_nms_top_n_test=2000,
                                                                 box_detections_per_img=500)
    # Adjust the box predictor for 200 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if hasattr(model, 'roi_heads'):
        model.roi_heads.score_thresh = 0.01
        model.roi_heads.nms_thresh = 0.5
        model.roi_heads.detections_per_img = 500

    model.to(device)

    print('Setup Training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_single_epoch(model, optimizer, dataloader_train, device, epoch, scaler)
        lr_scheduler.step()
        evaluate(model, dataloader_val, device)
        save_checkpoint(model, optimizer, lr_scheduler, scaler, epoch, args.model_out_dir)

    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: ", training_time)
    save_checkpoint(model, optimizer, lr_scheduler, scaler, args.epochs, args.model_out_dir)


def generate_rollback_model_name(out_dir, epoch):
    return out_dir + f"/model_epoch_{epoch}.pth"

def generate_model_name(out_dir):
    return out_dir + f'/latest_model.pth'

def save_checkpoint(model, optimizer, schedular, scaler, epoch, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': schedular.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }

    if epoch % 5 == 0:
        rollback_name = generate_rollback_model_name(out_dir, epoch)
        torch.save(checkpoint, rollback_name)
        print(f"Rollback checkpoint for epoch {epoch} saved to {rollback_name}.")

    filename = generate_model_name(out_dir)
    torch.save(checkpoint, filename)
    print(f"Latest checkpoint saved to {filename}")


def load_checkpoint(resume_path):
    if not os.path.exists(resume_path):
        print(f"Latest checkpoint not found, training from scratch.")
        return None

    checkpoint = torch.load(resume_path)
    return checkpoint


local_train_preset = argparse.Namespace(
    train_size=20,
    val_size=10,
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=10,
    epochs=10,
    milestones=[100, 150],
    gamma=0.1,
    device='cpu',
    model_out_dir='model',
    resume='model/latest_model.pth',
    start_epoch=0,
)

if __name__ == '__main__':
    main(local_train_preset)
