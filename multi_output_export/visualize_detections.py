import os
import sys
import random
import cv2
import numpy as np

# Ensure project root is in sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    print("=" * 70)
    print("Visual Detection Validator Generator")
    print("=" * 70)

    classes_path = os.path.join(_PROJECT_ROOT, "class.txt")
    class_names = load_classes(classes_path)

    images_dir = os.path.join(_PROJECT_ROOT, "multi_output_export", "dataset_export", "images")
    labels_dir = os.path.join(_PROJECT_ROOT, "multi_output_export", "dataset_export", "labels")
    output_dir = os.path.join(_PROJECT_ROOT, "multi_output_export", "dataset_export", "validation_visuals")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("ERROR: Exported dataset not found. Run process_dataset.py first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # List all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    if not label_files:
        print("ERROR: No label files found in labels directory.")
        return

    # Select a sample of 20 random files to visualize
    sample_size = min(20, len(label_files))
    sampled_labels = random.sample(label_files, sample_size)

    print(f"Generating visual validation for {sample_size} random frames...")

    # Color palette for classes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

    for label_file in sampled_labels:
        base_name = os.path.splitext(label_file)[0]
        img_name = f"{base_name}.jpg"
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        
        # Read labels
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx = float(parts[1]) * w
            cy = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h

            x1 = int(cx - box_w / 2.0)
            y1 = int(cy - box_h / 2.0)
            x2 = int(cx + box_w / 2.0)
            y2 = int(cy + box_h / 2.0)

            # Clamp coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            class_label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            color = [int(c) for c in colors[class_id % len(colors)]]

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw text
            text = class_label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Text background
            cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0] + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save visualized image
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)

    print(f"SUCCESS: Visual validation frames generated in: {output_dir}")

if __name__ == "__main__":
    main()
