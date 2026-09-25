import os
import sys
from PIL import Image

from model import MODEL_CLASSES, predict_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_prediction.py path/to/image.jpg")
        print("Example: python validate_prediction.py D:/BONE_MARROW_DS/BLA/BLA_00023.jpg")
        return 1

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1

    try:
        image = Image.open(image_path).convert("RGB")
        probs = predict_image(image)
        predicted_index = int(probs.argmax().item())
        predicted_class = MODEL_CLASSES[predicted_index]
        confidence = float(probs[predicted_index].item() * 100)

        print(f"Image: {image_path}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("All class probabilities:")

        for name, prob in zip(MODEL_CLASSES, probs.tolist()):
            print(f"  {name}: {float(prob * 100):.2f}%")

        return 0
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
