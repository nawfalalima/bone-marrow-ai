import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from model import MODEL, MODEL_CLASSES, MODEL_LOAD_ERROR, predict_image

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Please upload an image."})

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({"success": False, "error": "Please upload an image."})

    file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "error": "Please upload a JPG, JPEG, or PNG image."})

    if MODEL is None:
        return jsonify(
            {
                "success": False,
                "error": (
                    "The model checkpoint is missing or unavailable. "
                    "Add the trained model file to model/bone_marrow_resnet18.pth "
                    "before running inference."
                ),
            }
        )

    try:
        image = Image.open(uploaded_file.stream)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        probabilities = predict_image(image)

        class_probabilities = {
            class_name: round(float(probability * 100), 2)
            for class_name, probability in zip(MODEL_CLASSES, probabilities.tolist())
        }

        predicted_index = int(probabilities.argmax().item())
        predicted_class = MODEL_CLASSES[predicted_index]
        confidence = round(float(probabilities[predicted_index].item() * 100), 2)

        return jsonify(
            {
                "success": True,
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": class_probabilities,
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": f"Unable to process the image: {str(exc)}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
