import os
import csv
import xml.etree.ElementTree as ET
from PIL import Image

def load_labels_map(labels_path: str) -> dict:
    labels_map = {}
    with open(labels_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            labels_map[row[0]] = int(row[1])
    return labels_map

def extract_crops(image_dir: str,
                  annotation_dir: str,
                  labels_path: str,
                  out_dir: str,
                  padding: float = 0.12,
                  crop_size: int = 128):
    """
    Extracts and saves individual piece crops from full images.
    Output structure: out_dir/<class_id>/<image_name>_<box_idx>.png
    Ready for torchvision ImageFolder.
    """
    labels_map = load_labels_map(labels_path)
    os.makedirs(out_dir, exist_ok=True)

    filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.png')]
    total_crops = 0

    for filename in filenames:
        image_path = os.path.join(image_dir, f"{filename}.png")
        annotation_path = os.path.join(annotation_dir, f"{filename}.xml")

        image = Image.open(image_path).convert('RGB')
        img_w, img_h = 640, 640

        tree = ET.parse(annotation_path)

        for box_idx, obj in enumerate(tree.findall('object')):
            name = obj.find('name').text
            class_id = labels_map[name]

            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)

            # Apply padding
            bw = xmax - xmin
            bh = ymax - ymin
            xmin = max(0, xmin - bw * padding)
            ymin = max(0, ymin - bh * padding)
            xmax = min(img_w, xmax + bw * padding)
            ymax = min(img_h, ymax + bh * padding)

            crop = image.crop((xmin, ymin, xmax, ymax))
            crop = crop.resize((crop_size, crop_size), Image.BILINEAR)

            class_dir = os.path.join(out_dir, str(class_id))
            os.makedirs(class_dir, exist_ok=True)

            out_path = os.path.join(class_dir, f"{filename}_{box_idx}.png")
            crop.save(out_path)
            total_crops += 1

    print(f"Extracted {total_crops} crops to {out_dir}")