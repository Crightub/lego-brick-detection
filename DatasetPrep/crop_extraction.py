import csv
import os
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
                  crop_size: int = 128,
                  min_window_size: int = 48):  # Added minimum window
    """
    Extracts and saves individual piece crops from full images.
    Output structure: out_dir/<class_id>/<image_name>_<box_idx>.png
    """
    # Assuming load_labels_map is defined elsewhere in your code
    labels_map = load_labels_map(labels_path)
    os.makedirs(out_dir, exist_ok=True)

    filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.png')]
    total_crops = 0

    for filename in filenames:
        image_path = os.path.join(image_dir, f"{filename}.png")
        annotation_path = os.path.join(annotation_dir, f"{filename}.xml")

        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size  # Dynamically get size just in case

        tree = ET.parse(annotation_path)

        for box_idx, obj in enumerate(tree.findall('object')):
            name = obj.find('name').text
            class_id = labels_map[name]

            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)

            bw = xmax - xmin
            bh = ymax - ymin

            cx = xmin + (bw / 2)
            cy = ymin + (bh / 2)

            max_dim = max(bw, bh)

            window_size = max_dim * (1 + (padding * 2))
            window_size = max(window_size, min_window_size)

            new_xmin = int(cx - (window_size / 2))
            new_ymin = int(cy - (window_size / 2))
            new_xmax = int(cx + (window_size / 2))
            new_ymax = int(cy + (window_size / 2))

            crop = image.crop((new_xmin, new_ymin, new_xmax, new_ymax))

            crop = crop.resize((crop_size, crop_size), getattr(Image, 'Resampling', Image).LANCZOS)

            class_dir = os.path.join(out_dir, f"{int(class_id):03d}")
            os.makedirs(class_dir, exist_ok=True)

            out_path = os.path.join(class_dir, f"{filename}_{box_idx}.png")
            crop.save(out_path)
            total_crops += 1

    print(f"Extracted {total_crops} crops to {out_dir}")
