import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image

# ---------------------------------
# App Initialization
# ---------------------------------

app = FastAPI(title="Smart Retail AI API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "deployment_files")

print("Loading models...")

# Load CNN model
vision_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "ripeness_model.keras")
)

# Load regression models
spoilage_model = joblib.load(
    os.path.join(MODEL_DIR, "spoilage_model.pkl")
)

pricing_model = joblib.load(
    os.path.join(MODEL_DIR, "pricing_model.pkl")
)

# Load metadata
with open(os.path.join(MODEL_DIR, "ripeness_metadata.json")) as f:
    ripeness_metadata = json.load(f)

class_names = ripeness_metadata["class_names"]
IMG_SIZE = ripeness_metadata.get("img_size", 224)

print("Models loaded successfully.")


# ---------------------------------
# Utility: Image Preprocessing
# ---------------------------------

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32")
    image = preprocess_input(image)
    return image


# ---------------------------------
# Health Check Endpoint
# ---------------------------------

@app.get("/")
def health_check():
    return {"status": "API is running successfully"}


## TTA
def predict_with_tta(image):
    
    # Original
    img1 = preprocess_image(image)

    # Horizontal flip
    img2 = preprocess_image(image.transpose(Image.FLIP_LEFT_RIGHT))

    # Slight rotation
    img3 = preprocess_image(image.rotate(10))
    img4 = preprocess_image(image.rotate(-10))

    images = np.stack([img1, img2, img3, img4], axis=0)

    preds = vision_model.predict(images, verbose=0)

    # Average predictions
    final_pred = np.mean(preds, axis=0)

    return final_pred

# ---------------------------------
# Prediction Endpoint
# ---------------------------------


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    quantity: int = Form(...),
    sales_velocity: float = Form(...),
    days_in_storage: int = Form(...)
):

    # -------- Layer 1: Vision --------
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = preprocess_image(image)

    preds = predict_with_tta(image)
    stage_index = int(np.argmax(preds))
    stage_label = class_names[stage_index]
    confidence = float(np.max(preds))

    # Confidence gate
    CONFIDENCE_THRESHOLD = 0.75

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "Low Confidence – Manual Review Required",
            "predicted_stage": stage_label,
            "confidence": confidence
        }

    # -------- Layer 2: Spoilage --------
    features = pd.DataFrame([{
        "ripeness_stage": stage_index,
        "temperature": temperature,
        "humidity": humidity,
        "quantity": quantity,
        "sales_velocity": sales_velocity,
        "days_in_storage": days_in_storage
    }])

    spoilage_prob, expected_waste = spoilage_model.predict(features)[0]

    # -------- Layer 3: Pricing --------
    pricing_features = pd.DataFrame([{
        "spoilage_probability": spoilage_prob,
        "expected_waste_kg": expected_waste,
        "quantity": quantity,
        "sales_velocity": sales_velocity
    }])

    optimal_discount = int(pricing_model.predict(pricing_features)[0])

    return {
        "status": "Prediction Successful",
        "ripeness_stage": stage_label,
        "confidence": confidence,
        "spoilage_probability": float(spoilage_prob),
        "expected_waste_kg": float(expected_waste),
        "optimal_discount_percent": optimal_discount
    }