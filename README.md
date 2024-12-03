# deep-learning-image-segmentation-tensorflow

This project performs image segmentation using TensorFlow 2 and a U-Net model architecture. It includes data preprocessing, model building, training, evaluation, and API deployment using Flask.

## Directory Structure

```bash
image_segmentation_project/
├── README.md
├── requirements.txt
├── saved_model.keras
├── saved_images/
├── data/
│   ├── images/
│   └── annotations/
│   └── preprocessed/
├── scripts/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
```

README.md: Project documentation.
requirements.txt: List of Python dependencies.
data/: Directory containing images and annotations.
scripts/: Contains all the Python scripts for the project.
