import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image


def export_yolo(image_path, bboxes, class_names, output_dir):
    img = Image.open(image_path)
    img_w, img_h = img.size

    label_file = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    with open(label_file, 'w') as f:
        for det in bboxes:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            class_id = class_names.index(det['label']) if det['label'] in class_names else 0
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    img.save(os.path.join(output_dir, os.path.basename(image_path)))


def export_voc(image_path, bboxes, class_names, output_dir):
    img = Image.open(image_path)
    img_w, img_h = img.size

    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.basename(output_dir)
    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_w)
    height = ET.SubElement(size, 'height')
    height.text = str(img_h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    for det in bboxes:
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = det['label']
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(det['bbox'][0]))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(det['bbox'][1]))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(det['bbox'][2]))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(det['bbox'][3]))

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    xml_file = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
    with open(xml_file, 'w') as f:
        f.write(xml_str)
    img.save(os.path.join(output_dir, os.path.basename(image_path)))


def export_coco(image_path, bboxes, class_names, output_dir, image_id=1):
    img = Image.open(image_path)
    img_w, img_h = img.size

    coco_data = {
        "images": [{
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": img_w,
            "height": img_h
        }],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }

    ann_id = 1
    for det in bboxes:
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1
        h = y2 - y1
        category_id = class_names.index(det['label']) if det['label'] in class_names else 0
        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1

    json_file = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')
    with open(json_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    img.save(os.path.join(output_dir, os.path.basename(image_path)))
