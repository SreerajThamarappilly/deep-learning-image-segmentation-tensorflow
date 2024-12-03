# scripts/train.py

import os
import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from model import unet_model

def data_generator(images_dir, masks_dir, batch_size):
    """
    Yields batches of resized images and masks for training.

    Args:
        images_dir (str): Directory containing preprocessed images.
        masks_dir (str): Directory containing preprocessed masks.
        batch_size (int): Number of samples per batch.

    Yields:
        Tuple[np.array, np.array]: Batch of resized images and masks.
    """
    image_ids = os.listdir(images_dir)
    while True:
        np.random.shuffle(image_ids)
        for i in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[i:i+batch_size]
            batch_images = []
            batch_masks = []
            for id_ in batch_ids:
                # Load image and mask
                image = np.load(os.path.join(images_dir, id_))
                mask = np.load(os.path.join(masks_dir, id_))

                # Resize image and mask to match the model's input size
                resized_image = cv2.resize(image, (256, 256))
                resized_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

                # Normalize image and append to the batch
                batch_images.append(resized_image / 255.0)
                batch_masks.append(resized_mask)

            yield np.array(batch_images), np.expand_dims(np.array(batch_masks), axis=-1)

if __name__ == '__main__':
    images_dir = 'data/preprocessed/images'
    masks_dir = 'data/preprocessed/masks'

    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    batch_size = 8
    steps_per_epoch = len(os.listdir(images_dir)) // batch_size

    checkpoint = ModelCheckpoint('saved_model.keras', monitor='loss', verbose=1, save_best_only=True, save_weights_only=False)

    model.fit(
        data_generator(images_dir, masks_dir, batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        callbacks=[checkpoint]
    )
