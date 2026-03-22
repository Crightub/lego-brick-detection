import torch
from PIL import Image, ImageDraw, ImageFont
from pipeline import LegoPipeline


def visualize(image: Image.Image, result: dict, output_path: str):
    """Draws boxes and part IDs onto the image and saves it."""
    draw = ImageDraw.Draw(image)

    boxes  = result['boxes']
    labels = result['labels']
    scores = result['scores']

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box.tolist()
        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
        draw.text((xmin, ymin - 12), f"ID:{label} {score:.2f}", fill='red')

    image.save(output_path)
    print(f"Saved visualisation to {output_path}")


if __name__ == '__main__':
    pipeline = LegoPipeline(
        stage1_path='model/stage1/weights/best.pt',
        stage2_path='model/stage2/best_model.pth',
        labels_map_path='data/labels_map.csv',
        device='cpu',
        conf=0.01,
        iou=0.3,
    )

    image = Image.open('data/train/images/1.png').convert('RGB')
    result = pipeline.predict(image)

    print(f"Detected {len(result['boxes'])} pieces")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  Part ID: {label.item():>4}  score: {score.item():.4f}")

    visualize(image.copy(), result, 'output.png')