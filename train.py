import argparse
import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from b200_lego_dataset import load_train_set, load_val_set
from engine import train_single_epoch
from eval import evaluate


def main(args):
    device = torch.device(args.device)
    print('Using device:', device)

    # Load train and validation data set
    print('Loading data...')
    dataset_train = load_train_set(args.train_size)
    dataset_val = load_val_set(args.val_size)

    num_classes = dataset_train.label_count()
    print(f'Number of classes: {num_classes}')

    # create dataloaders
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

    model.to(device)

    print('Setup Training...')
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # run epochs
    start_time = time.time()
    for epoch in range(args.epochs):
        train_single_epoch(model, optimizer, dataloader_train, device, epoch)
        lr_scheduler.step()
        evaluate(model, dataloader_val)

    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: ", training_time)

    save_checkpoint(model, optimizer, args.epochs, args.model_out_dir)


def generate_model_name(out_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    return out_dir + '/' + timestamp + '.pth'


def save_checkpoint(model, optimizer, epoch, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = generate_model_name(out_dir)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


local_train_preset = argparse.Namespace(
    train_size=15,
    val_size=5,
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=5,
    epochs=1,
    milestones=[100, 150],
    gamma=0.1,
    device='cpu',
    model_out_dir='model'
)

if __name__ == '__main__':
    main(local_train_preset)
