# scripts/evaluate.py

import os
import numpy as np
import tensorflow as tf
import cv2  # For resizing

def data_generator(images_dir, masks_dir, batch_size):
    """
    Generates batches of resized images and masks for evaluation.

    Args:
        images_dir (str): Directory containing preprocessed images.
        masks_dir (str): Directory containing preprocessed masks.
        batch_size (int): Number of samples per batch.

    Yields:
        Tuple[np.array, np.array]: Batch of resized images and masks.
    """
    image_ids = os.listdir(images_dir)
    while True:
        for i in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[i:i+batch_size]
            batch_images = []
            batch_masks = []
            for id_ in batch_ids:
                # Load preprocessed image and mask
                image = np.load(os.path.join(images_dir, id_))
                mask = np.load(os.path.join(masks_dir, id_))

                # Resize image and mask to match the model's expected input size
                resized_image = cv2.resize(image, (256, 256))  # Resize to 256x256
                resized_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)  # Resize to 256x256

                # Normalize image and expand mask dimensions
                batch_images.append(resized_image / 255.0)  # Normalize image
                batch_masks.append(resized_mask)

            yield np.array(batch_images), np.expand_dims(np.array(batch_masks), axis=-1)  # Add channel dimension for masks

if __name__ == '__main__':
    images_dir = 'data/preprocessed/images'
    masks_dir = 'data/preprocessed/masks'

    # Load the entire model
    model = tf.keras.models.load_model('saved_model.keras')

    batch_size = 8
    steps = len(os.listdir(images_dir)) // batch_size

    loss, accuracy = model.evaluate(
        data_generator(images_dir, masks_dir, batch_size),
        steps=steps
    )
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
