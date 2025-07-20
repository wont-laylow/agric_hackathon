# =============================================================================
# FILE: app/ml/inference.py
# VERSION: CORRECTED - Fixes the CLIP gatekeeper bug.
# =============================================================================
import joblib
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import timm
import io
import clip

from ..core.config import settings

# =============================================================================
# REQUIRED DEFINITIONS FOR JOBLIB (No changes here)
# =============================================================================
def convert_to_pil(image: np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image

class MobileNetV3CropModel(nn.Module):
    def __init__(self, model_name=settings.MODEL_NAME, num_classes=len(settings.CLASS_NAMES), pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.backbone.num_features
        self.backbone.reset_classifier(0)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(num_features),
            nn.Dropout(0.2), nn.Linear(num_features, num_features // 4),
            nn.ReLU(inplace=True), nn.BatchNorm1d(num_features // 4),
            nn.Dropout(0.1), nn.Linear(num_features // 4, num_classes)
        )
    def forward(self, x):
        features = self.backbone.forward_features(x)
        out = self.classifier(features)
        return out

class MobileNetEnsemble:
    def __init__(self, model_paths, device):
        self.models = []
        self.device = device

class PredictionPipeline:
    def __init__(self, model, transforms, idx_to_class):
        self.device = torch.device('cpu')
        self.model = model
        for m in self.model.models:
            m.to(self.device)
            m.eval()
        self.transforms = transforms
        self.idx_to_class = idx_to_class

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================
ml_models: Dict[str, Any] = {}
# --- [FIX #1] --- FORCE CPU USAGE FOR RELIABILITY ---
# This avoids all potential CUDA driver issues on local machines and servers.
device = "cpu"
print(f"INFO:     Forcing all ML models to use device: '{device}'")

def load_models(model_path: Path):
    """Loads BOTH your specialist pipeline and the CLIP gatekeeper model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Specialist model file not found at {model_path}")

    # Load specialist model (no changes needed here)
    print(f"INFO:     Loading specialist model from: {model_path}")
    sys.modules['__main__'].convert_to_pil = convert_to_pil
    sys.modules['__main__'].MobileNetV3CropModel = MobileNetV3CropModel
    sys.modules['__main__'].MobileNetEnsemble = MobileNetEnsemble
    sys.modules['__main__'].PredictionPipeline = PredictionPipeline
    pipeline = joblib.load(model_path)
    del sys.modules['__main__'].convert_to_pil, sys.modules['__main__'].MobileNetV3CropModel, sys.modules['__main__'].MobileNetEnsemble, sys.modules['__main__'].PredictionPipeline
    ml_models["mobilenet_pipeline"] = pipeline
    print("INFO:     Specialist model loaded successfully.")

    # Load CLIP gatekeeper model
    if "gatekeeper" not in ml_models:
        print("INFO:     Loading CLIP Gatekeeper model...")
        # The 'device' variable is now hardcoded to 'cpu'
        gatekeeper_model, gatekeeper_preprocess = clip.load("ViT-B/32", device=device)
        ml_models["gatekeeper"] = gatekeeper_model
        ml_models["gatekeeper_preprocess"] = gatekeeper_preprocess
        print("INFO:     CLIP Gatekeeper model loaded successfully.")

def unload_models():
    ml_models.clear()
    print("INFO:     All models unloaded.")

# =============================================================================
# THE CORRECTED GATEKEEPER FUNCTION
# =============================================================================
def is_image_a_leaf_permissive(image_bytes: bytes, debug=True) -> bool:
    """
    Uses CLIP with improved prompts and strict, reliable logic to validate images.
    """
    gatekeeper = ml_models.get("gatekeeper")
    preprocess = ml_models.get("gatekeeper_preprocess")
    if not gatekeeper or not preprocess:
        print("WARNING: Gatekeeper model not loaded. Rejecting image for safety.")
        return False

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # The .to(device) call will now reliably work because device is 'cpu'
        image = preprocess(pil_image).unsqueeze(0).to(device)

        plant_prompts = [
            "a close-up photo of a single plant leaf", "a diseased leaf from a farm crop",
            "foliage, vegetation, or plant life", "a leaf showing signs of blight, rust, or spots"
        ]
        non_plant_prompts = [
            "a photo of a person, a human face, or an animal", "a car, building, or street scene",
            "a picture of a computer screen, a document, or text", "a blurry, out-of-focus, or dark image",
            "a drawing or cartoon"
        ]

        all_prompts = plant_prompts + non_plant_prompts
        text_tokens = clip.tokenize(all_prompts).to(device)

        with torch.no_grad():
            logits_per_image, _ = gatekeeper(image, text_tokens)
            probabilities = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        total_plant_score = sum(probabilities[:len(plant_prompts)])
        total_non_plant_score = sum(probabilities[len(plant_prompts):])

        is_plant_dominant = total_plant_score > (total_non_plant_score + 0.20)
        is_plant_confident = total_plant_score > 0.60
        is_acceptable = is_plant_dominant and is_plant_confident

        if debug:
            print(f"DEBUG (Gatekeeper): Plant Score: {total_plant_score:.4f} | Non-Plant Score: {total_non_plant_score:.4f}")
            print(f"DEBUG (Gatekeeper): Is Dominant? -> {is_plant_dominant} | Is Confident? -> {is_plant_confident}")
            print(f"DEBUG (Gatekeeper): FINAL DECISION: {'ACCEPT' if is_acceptable else 'REJECT'}")

        return is_acceptable

    except Exception as e:
        # --- [FIX #2] --- FAIL-CLOSED MECHANISM ---
        # If any error occurs, REJECT the image instead of allowing it through.
        print(f"ERROR: Gatekeeper check failed with an unexpected exception: {e}. REJECTING image for safety.")
        return False

# =============================================================================
# SPECIALIST MODEL PREDICTION (No changes here)
# =============================================================================
def run_prediction(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Runs your specialist model to get top-5 disease predictions.
    """
    pipeline = ml_models.get("mobilenet_pipeline")
    if not pipeline:
        raise ValueError("Specialist model is not loaded.")

    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = pipeline.transforms(pil_image).unsqueeze(0).to(pipeline.device)

        with torch.no_grad():
            batch_predictions = [m(tensor) for m in pipeline.model.models]
            ensemble_pred = torch.stack(batch_predictions).mean(dim=0)
            all_probabilities = F.softmax(ensemble_pred, dim=1)[0]
            top5_probs, top5_idx = torch.topk(all_probabilities, 5)

        results = []
        for i in range(len(top5_probs)):
            confidence = top5_probs[i].item() * 100
            results.append({
                "predicted_class": pipeline.idx_to_class.get(top5_idx[i].item(), "Unknown"),
                "confidence": confidence
            })

        print(f"INFO: Top prediction: {results[0]['predicted_class']} ({results[0]['confidence']:.2f}%)")
        return results

    except Exception as e:
        print(f"ERROR: Specialist prediction failed: {e}")
        raise