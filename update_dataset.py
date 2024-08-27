import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
from cone_utils import *

source_dir = '/source/dataset/path/SOURCE_DATASET_NAME'
target_dir = '/target/dataset/path/TARGET_DATASET_NAME'
# scratch_dir = '/scratch/directory/path/for/visualization'

resize_imgs = True
target_res = (640, 640)

# Create necessary directories
for subdir in ['train', 'valid']:
    os.makedirs(os.path.join(target_dir, subdir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, subdir, 'labels'), exist_ok=True)
os.makedirs(target_dir, exist_ok=True)
# os.makedirs(scratch_dir, exist_ok=True)

ideal_hues = {
    'yellow': 30,
    'blue': 100,
}

def process_images_and_labels(set_name):
    image_dir = os.path.join(source_dir, set_name, 'images')
    label_dir = os.path.join(source_dir, set_name, 'labels')
    images = os.listdir(image_dir)

    for image_name in tqdm(images, desc=f'Processing {set_name} images'):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))
        image = Image.open(image_path)
        image_np = np.array(image)
        image_width, image_height = image.size

        # Resize Image to target resolution and save the resized image
        if resize_imgs:
            image_resized = image.resize(target_res)   
            target_path = os.path.join(target_dir, set_name, 'images', image_name)
            image_resized.save(target_path)
        else:
            # Copy image to new directory
            shutil.copy(image_path, os.path.join(target_dir, set_name, 'images', image_name))

        with open(label_path, 'r') as file:
            labels = file.readlines()

        new_labels = []
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.imshow(image)

        for label in labels:
            parts = label.strip().split()
            coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
            pixel_coords = np.stack([coords[:,0] * image_width, coords[:,1] * image_height], axis=-1).astype(np.int32)

            sampled_points = sample_points_within_polygon(pixel_coords, num_samples=30)
            average_hue = calculate_average_hue(sampled_points, image_np)
            closest_color = calculate_closest_hue(average_hue, ideal_hues)
            class_id = 0 if closest_color == 'yellow' else 1
            new_labels.append(f"{class_id} {' '.join([f'{x:.6f}' for x in coords.flatten()])}\n")

            # # Plot polygon and text
            # polygon = Polygon(pixel_coords, closed=True, edgecolor='red', fill=None, linewidth=2)
            # ax.add_patch(polygon)
            # centroid = pixel_coords.mean(axis=0)
            # ax.text(centroid[0], centroid[1], closest_color, color='black', fontsize=12, ha='center', va='center')

        # Save new labels
        new_label_path = os.path.join(target_dir, set_name, 'labels', image_name.replace('.jpg', '.txt'))
        with open(new_label_path, 'w') as file:
            file.writelines(new_labels)

        # ax.set_axis_off()
        # fig.tight_layout()
        # plt.savefig(os.path.join(scratch_dir, image_name.replace('.jpg', '_annotated.jpg')))
        # plt.close()

process_images_and_labels('train')
process_images_and_labels('valid')

# Update data.yaml
data_yaml_content = """names:
- Yellow
- Blue
nc: 2
train: {target_dir}/train/images
val: {target_dir}/valid/images
"""
with open(os.path.join(target_dir, 'data.yaml'), 'w') as file:
    file.write(data_yaml_content.format(target_dir=target_dir))