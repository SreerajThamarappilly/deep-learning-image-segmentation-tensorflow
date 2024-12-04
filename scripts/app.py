# scripts/app.py

from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import uuid

app = Flask(__name__)

# Load the trained model
model = load_model('saved_model.keras')

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Receives an image file via POST request and returns the segmentation mask.
    Saves both input and output images to a directory within the project.
    """
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Read the image file
        image = Image.open(file)
        original_image = image.copy()  # Keep a copy to save later

        # Resize and preprocess the image
        image = image.resize((256, 256))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict the mask
        mask = model.predict(image_array)[0]

        # Convert mask to binary
        mask = (mask > 0.5).astype(np.uint8)

        # Create a directory to save images if it doesn't exist
        save_dir = 'saved_images'
        os.makedirs(save_dir, exist_ok=True)

        # Generate a unique filename
        filename = str(uuid.uuid4())

        # Save the original input image
        input_image_path = os.path.join(save_dir, f'{filename}_input.jpg')
        original_image.save(input_image_path)

        # Save the mask image
        mask_image = Image.fromarray(mask.squeeze() * 255)
        output_image_path = os.path.join(save_dir, f'{filename}_mask.png')
        mask_image.save(output_image_path)

        # Encode the mask to base64 for the response
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'mask': mask_base64}), 200
    else:
        return jsonify({'error': 'File processing error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
