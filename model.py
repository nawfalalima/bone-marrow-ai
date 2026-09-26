import os
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch import nn
from torchvision import models, transforms

MODEL_CLASSES = ["BLA", "EOS", "MON", "NGS", "PLM"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "bone_marrow_resnet18.pth"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
MODEL_DOWNLOAD_URL = os.environ.get("MODEL_DOWNLOAD_URL") or os.environ.get("MODEL_URL")


def ensure_model_available(model_path=MODEL_PATH, download_url=MODEL_DOWNLOAD_URL):
    if model_path.exists():
        return model_path

    if download_url:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(download_url, model_path)
        if model_path.exists():
            return model_path

    raise FileNotFoundError(
        f"Model checkpoint not found at: {model_path}. "
        "Set MODEL_PATH or MODEL_DOWNLOAD_URL to a valid checkpoint before running inference."
    )


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(MODEL_CLASSES))
    return model


def load_model(model_path=MODEL_PATH):
    model_path = ensure_model_available(model_path=model_path)

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Could not load the model checkpoint from {model_path}: {exc}") from exc

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint format: expected a PyTorch state_dict-like object, got {type(checkpoint)}."
        )

    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model = build_model()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint format is incompatible with the expected ResNet18 state_dict. "
            f"Original load error: {exc}"
        ) from exc

    model.eval()
    return model


MODEL = None
MODEL_LOAD_ERROR = None

try:
    MODEL = load_model()
except Exception as exc:  # pragma: no cover - behavior is deployment-safe when model is absent
    MODEL_LOAD_ERROR = str(exc)


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)


def predict_image(image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        logits = MODEL(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    return probabilities.cpu()
