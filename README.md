# Bone Marrow Image Classifier

This project is a Flask web application that uploads a microscopy image and predicts the bone marrow class using a PyTorch ResNet model.

## Features
- Upload JPG, JPEG, or PNG files
- Run inference from a trained model
- Display predicted class and confidence
- View probability breakdown for each class

## Setup

1. Create a virtual environment:
   python -m venv .venv
2. Activate it:
   - Windows: .venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Start the app:
   python app.py
5. Open the app in your browser at:
   http://localhost:5000

## Model file

Place your trained weights file in the following location before running the app:

model/bone_marrow_resnet18.pth

If you are deploying to Render, add the model file to the repository or set the `MODEL_DOWNLOAD_URL` environment variable to a public downloadable URL for the `.pth` file.

The project is configured to use `MODEL_PATH` and `MODEL_DOWNLOAD_URL` so it can load a checkpoint in local development or in deployment environments without failing silently.
