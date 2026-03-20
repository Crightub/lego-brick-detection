import os
import xml.etree.ElementTree as ET

def convert_annotations_to_yolo(annotation_dir: str, yolo_out_dir: str):
    """
    Reads Pascal VOC XML annotations and writes YOLO binary format .txt files.
    All classes collapsed to class_id=0 ("piece").
    """
    os.makedirs(yolo_out_dir, exist_ok=True)

    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(annotation_dir, xml_file))
        root = tree.getroot()

        img_w = 640.0
        img_h = 640.0

        lines = []
        for obj in root.findall('object'):
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)

            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h
            w  = (xmax - xmin) / img_w
            h  = (ymax - ymin) / img_h

            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        out_name = os.path.splitext(xml_file)[0] + '.txt'
        with open(os.path.join(yolo_out_dir, out_name), 'w') as f:
            f.write('\n'.join(lines))

    print(f"Converted {len(os.listdir(yolo_out_dir))} annotation files to {yolo_out_dir}")