import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
import cv2

# False-positive mining exports use a single class so every box trains as "person".
PERSON_FP_CLASS = "person"


def detections_as_person_labels(detections):
    """Each exported object uses class name person (bbox/conf/track_id preserved)."""
    return [{**det, "label": PERSON_FP_CLASS} for det in detections]


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


def export_false_positive_frames(fp_frame_data, output_dir, class_names, format_type='yolo'):
    """
    Export all marked false positive frames with detections for model retraining.
    
    Args:
        fp_frame_data: Dict {frame_id: {detections, frame_image, timestamp}}
        output_dir: Base output directory
        class_names: Kept for API compatibility; every box is written as class "person".
        format_type: 'yolo', 'voc', or 'coco'
    
    Returns:
        Dict with export results {exported_frames, exported_detections, output_dir}
    """
    # Create subdirectories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    exported_frames = 0
    exported_detections = 0
    person_class_names = [PERSON_FP_CLASS]
    metadata = {
        'total_frames': len(fp_frame_data),
        'exported_frames': 0,
        'format': format_type,
        'class_names': person_class_names,
        'frames': []
    }
    
    # Process each marked frame
    for frame_id, frame_data in sorted(fp_frame_data.items()):
        # Skip frames without actual image data (manually entered frame numbers)
        if frame_data.get('frame_image') is None:
            continue
        
        frame_image = frame_data['frame_image']
        detections = detections_as_person_labels(frame_data.get('detections', []))
        
        # Save frame image
        image_filename = f"frame_{frame_id:06d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        cv2.imwrite(image_path, frame_image)
        
        # Save annotations based on format
        if format_type == 'yolo':
            _export_frame_yolo(image_path, detections, person_class_names, labels_dir, frame_id)
        elif format_type == 'voc':
            _export_frame_voc(image_path, detections, person_class_names, labels_dir, frame_id)
        elif format_type == 'coco':
            _export_frame_coco(image_path, detections, person_class_names, labels_dir, frame_id)
        
        # Update metadata
        metadata['frames'].append({
            'frame_id': frame_id,
            'image_file': image_filename,
            'num_detections': len(detections),
            'timestamp': frame_data.get('timestamp'),
            'classes': list(set([det['label'] for det in detections]))
        })
        
        exported_frames += 1
        exported_detections += len(detections)
    
    metadata['exported_frames'] = exported_frames
    metadata['exported_detections'] = exported_detections
    
    # Save metadata file
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'exported_frames': exported_frames,
        'exported_detections': exported_detections,
        'output_dir': output_dir
    }


def _export_frame_yolo(image_path, detections, class_names, labels_dir, frame_id):
    """Export single frame in YOLO format."""
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    label_filename = f"frame_{frame_id:06d}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    
    with open(label_path, 'w') as f:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            class_id = class_names.index(det['label']) if det['label'] in class_names else 0
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _export_frame_voc(image_path, detections, class_names, labels_dir, frame_id):
    """Export single frame in VOC format."""
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'false_positive_frames'
    filename = ET.SubElement(annotation, 'filename')
    filename.text = f"frame_{frame_id:06d}.jpg"
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_w)
    height = ET.SubElement(size, 'height')
    height.text = str(img_h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    
    for det in detections:
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
    xml_filename = f"frame_{frame_id:06d}.xml"
    xml_path = os.path.join(labels_dir, xml_filename)
    with open(xml_path, 'w') as f:
        f.write(xml_str)


def _export_frame_coco(image_path, detections, class_names, labels_dir, frame_id):
    """Export single frame in COCO format."""
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    coco_data = {
        "images": [{
            "id": frame_id,
            "file_name": f"frame_{frame_id:06d}.jpg",
            "width": img_w,
            "height": img_h
        }],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }
    
    ann_id = 1
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1
        h = y2 - y1
        category_id = class_names.index(det['label']) if det['label'] in class_names else 0
        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": frame_id,
            "category_id": category_id,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1
    
    json_filename = f"frame_{frame_id:06d}.json"
    json_path = os.path.join(labels_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
