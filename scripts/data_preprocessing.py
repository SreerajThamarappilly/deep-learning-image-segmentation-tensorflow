# scripts/data_preprocessing.py

import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

def load_annotations(annotation_file):
    """
    Loads and parses annotation data from a JSON file.

    Args:
        annotation_file (str): Path to the annotation JSON file.

    Returns:
        dict: Parsed annotations.
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def create_mask_from_annotation(image_annotations, image_shape):
    """
    Creates a segmentation mask from annotations for a single image.

    Args:
        image_annotations (list): List of annotation data for a single image.
        image_shape (tuple): Shape of the image (height, width, channels).

    Returns:
        np.array: Binary segmentation mask.
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for ann in image_annotations:
        # COCO 'segmentation' contains polygons
        if 'segmentation' in ann:
            for polygon in ann['segmentation']:
                polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [polygon_np], color=1)
    return mask

def preprocess_data(images_dir, annotations_file, output_dir):
    """
    Preprocesses the data and saves the preprocessed images and masks.

    Args:
        images_dir (str): Directory containing images.
        annotations_file (str): Path to the annotations JSON file.
        output_dir (str): Directory to save preprocessed data.
    """
    annotations_data = load_annotations(annotations_file)

    # Extract `images` and `annotations` sections
    images_info = annotations_data.get("images", [])
    annotations = annotations_data.get("annotations", [])

    for image_info in images_info:
        image_id = image_info['id']
        image_path = os.path.join(images_dir, image_info['file_name'])

        # Load the image
        image = img_to_array(load_img(image_path))

        # Filter annotations for this image
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        # Create a segmentation mask
        mask = create_mask_from_annotation(image_annotations, image.shape)

        # Save preprocessed image and mask
        output_image_path = os.path.join(output_dir, 'images', f"{image_id}.npy")
        output_mask_path = os.path.join(output_dir, 'masks', f"{image_id}.npy")
        np.save(output_image_path, image)
        np.save(output_mask_path, mask)

if __name__ == '__main__':
    images_dir = 'data/images'
    annotations_file = 'data/annotations/COCO_train_annos.json'
    output_dir = 'data/preprocessed'
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    preprocess_data(images_dir, annotations_file, output_dir)
