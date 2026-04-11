import argparse
from ultralytics import YOLO

def main(args):
    model = YOLO('../yolov8m.pt')  #

    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        lr0=args.lr,
        weight_decay=args.weight_decay,
        optimizer='Adam',
        device=args.device,
        project=args.model_out_dir,
        name='stage1',
        save_period=5,
        exist_ok=True,

        # Dense scene tuning — critical for your use case
        iou=0.3,               # low IoU NMS threshold — avoids merging adjacent pieces
        conf=0.01,             # low conf threshold during val — maximise recall
        max_det=500,           # matches your current detections_per_img

        mosaic=0.0,            # Turn off mosaic. Prevents stitching four 400-piece images together.
        scale=0.9,             # High scale variance (+/- 90%). Forces aggressive random zooming in/out and cropping.
        translate=0.3,         # Shifts the image up/down/left/right by 30%, cropping off dense edges.
        degrees=15.0,          # Adds slight rotation so the model doesn't overfit to Blender's perfect grid drops.
        perspective=0.001,     # Adds slight camera tilt/distortion to simulate real-world phone angles.
        erasing=0.2,
    )

def main_val(args):
    """Run validation only on a saved checkpoint."""
    model = YOLO(args.resume)
    metrics = model.val(
        data=args.data_yaml,
        imgsz=640,
        batch=args.batch_size,
        conf=0.01,
        iou=0.3,
        max_det=500,
        device=args.device,
    )
    print(f"Box mAP50:    {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")


local_train_preset = argparse.Namespace(
    data_yaml='YoloStage1/local_dataset.yaml',
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=8,
    epochs=1,
    device='cpu',              # GPU index — use 'cpu' for local testing
    model_out_dir='../model',
    resume=False,
)

if __name__ == '__main__':
    main(local_train_preset)