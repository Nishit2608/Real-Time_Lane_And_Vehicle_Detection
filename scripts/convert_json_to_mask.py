import os
import json
import numpy as np
import cv2
from tqdm import tqdm

def create_lane_mask(lanes, save_path, image_size=(720, 1280)):
    mask = np.zeros(image_size, dtype=np.uint8)

    for lane in lanes:
        if 'poly2d' in lane:
            for poly in lane['poly2d']:
                points = poly['vertices']
                points = [(int(x), int(y)) for x, y in points if x >= 0 and y >= 0]

                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(mask, points[i], points[i + 1], color=255, thickness=4)

    if np.count_nonzero(mask) > 0:
        cv2.imwrite(save_path, mask)

def process_json(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)

    for entry in tqdm(data, desc=f"Processing {json_path}"):
        image_name = entry['name'].replace('.jpg', '.png')
        lanes = [label for label in entry['labels'] if label['category'] == 'lane']
        save_path = os.path.join(output_dir, image_name)
        create_lane_mask(lanes, save_path)

# Set paths
train_json = R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k_labels_release\labels\bdd100k_labels_images_train.json"
val_json   = R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k_labels_release\labels\bdd100k_labels_images_val.json"

train_output = R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k\labels\lane_masks\train"
val_output   = R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k\labels\lane_masks\val"

# Run
process_json(train_json, train_output)
process_json(val_json, val_output)

print("✅ Mask generation complete!")
