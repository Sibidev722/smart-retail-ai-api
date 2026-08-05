import os
import io
import json
import logging

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("SmartRetailAI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "deployment_files")

RIPENESS_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ripeness_model.keras"
)

SPOILAGE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "spoilage_model.pkl"
)

PRICING_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pricing_model.pkl"
)

RIPENESS_METADATA_PATH = os.path.join(
    MODEL_DIR,
    "ripeness_metadata.json"
)

PREDICTION_METADATA_PATH = os.path.join(
    MODEL_DIR,
    "prediction_metadata.json"
)

PRICING_METADATA_PATH = os.path.join(
    MODEL_DIR,
    "pricing_metadata.json"
)


# ============================================================
# VERIFY MODEL FILES
# ============================================================

required_files = [
    RIPENESS_MODEL_PATH,
    SPOILAGE_MODEL_PATH,
    PRICING_MODEL_PATH,
    RIPENESS_METADATA_PATH
]

for path in required_files:
    if not os.path.exists(path):
        raise RuntimeError(
            f"Required deployment file not found: {path}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

logger.info("Loading Smart Retail AI models...")


try:

    # --------------------------------------------------------
    # Layer 1 - Ripeness CNN
    # --------------------------------------------------------

    vision_model = tf.keras.models.load_model(
        RIPENESS_MODEL_PATH
    )

    logger.info("Ripeness model loaded successfully.")


    # --------------------------------------------------------
    # Layer 2 - Spoilage model
    # --------------------------------------------------------

    spoilage_model = joblib.load(
        SPOILAGE_MODEL_PATH
    )

    logger.info("Spoilage model loaded successfully.")


    # --------------------------------------------------------
    # Layer 3 - Pricing model
    # --------------------------------------------------------

    pricing_model = joblib.load(
        PRICING_MODEL_PATH
    )

    logger.info("Pricing model loaded successfully.")


    # --------------------------------------------------------
    # Ripeness metadata
    # --------------------------------------------------------

    with open(
        RIPENESS_METADATA_PATH,
        "r"
    ) as f:

        ripeness_metadata = json.load(f)


    class_names = ripeness_metadata["class_names"]

    IMG_SIZE = int(
        ripeness_metadata.get(
            "img_size",
            224
        )
    )


    # --------------------------------------------------------
    # Optional metadata
    # --------------------------------------------------------

    prediction_metadata = {}

    if os.path.exists(PREDICTION_METADATA_PATH):

        with open(
            PREDICTION_METADATA_PATH,
            "r"
        ) as f:

            prediction_metadata = json.load(f)


    pricing_metadata = {}

    if os.path.exists(PRICING_METADATA_PATH):

        with open(
            PRICING_METADATA_PATH,
            "r"
        ) as f:

            pricing_metadata = json.load(f)


    logger.info(
        "Class names from training metadata: %s",
        class_names
    )

    logger.info(
        "Image size from training metadata: %s",
        IMG_SIZE
    )

    logger.info(
        "All models loaded successfully."
    )


except Exception as e:

    logger.exception(
        "Failed to initialize ML models."
    )

    raise RuntimeError(
        f"Model initialization failed: {e}"
    )


# ============================================================
# IMPORTANT:
# SPOILAGE MODEL RIPENESS ENCODING
# ============================================================

# Your Layer-2 model uses this semantic encoding.
#
# DO NOT use the CNN class index directly because the CNN
# class ordering comes from image_dataset_from_directory().
#
# We first get the CNN label, then convert that label into
# the value expected by the spoilage model.

LABEL_TO_STAGE = {
    "unripe": 0,
    "ripe": 1,
    "overripe": 2,
    "rotten": 3
}


# ============================================================
# FASTAPI INITIALIZATION
# ============================================================

app = FastAPI(
    title="Smart Retail AI API",
    description=(
        "Ripeness Detection → Spoilage Prediction "
        "→ Dynamic Pricing Recommendation"
    ),
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    IMPORTANT:

    This preprocessing intentionally follows the training
    pipeline of the uploaded notebook.

    We DO NOT call MobileNetV2 preprocess_input() here.

    We DO NOT divide pixels by 255 here.

    We resize the RGB image and convert it to float32,
    matching the input representation used during training.
    """

    # Ensure 3 channels
    image = image.convert("RGB")

    # Match training image size
    image = image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    # Convert to numpy
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    #
    # (224,224,3)
    #       ↓
    # (1,224,224,3)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# LAYER 1
# RIPENESS PREDICTION
# ============================================================

def predict_ripeness(image: Image.Image):

    image_array = preprocess_image(
        image
    )

    predictions = vision_model.predict(
        image_array,
        verbose=0
    )

    # predictions shape:
    #
    # (1, number_of_classes)

    probabilities = predictions[0]

    stage_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        np.max(probabilities)
    )

    stage_label = str(
        class_names[stage_index]
    )


    # Useful debugging
    probability_dict = {

        str(class_names[i]): round(
            float(probabilities[i]),
            6
        )

        for i in range(
            len(class_names)
        )
    }


    logger.info(
        "CNN probabilities: %s",
        probability_dict
    )

    logger.info(
        "Ripeness prediction: %s | confidence: %.4f",
        stage_label,
        confidence
    )


    return (
        stage_label,
        confidence,
        probability_dict
    )


# ============================================================
# LAYER 2
# SPOILAGE PREDICTION
# ============================================================

def predict_spoilage(
    stage_label: str,
    temperature: float,
    humidity: float,
    quantity: float,
    sales_velocity: float,
    days_in_storage: int
):

    normalized_label = (
        stage_label
        .strip()
        .lower()
    )


    if normalized_label not in LABEL_TO_STAGE:

        raise ValueError(
            f"Unknown ripeness label '{stage_label}'. "
            f"Expected one of {list(LABEL_TO_STAGE.keys())}"
        )


    # THIS is the value expected by Layer 2.
    #
    # unripe   -> 0
    # ripe     -> 1
    # overripe -> 2
    # rotten   -> 3

    numeric_stage = LABEL_TO_STAGE[
        normalized_label
    ]


    # Exact feature names/order used by spoilage model.

    features = pd.DataFrame(
        [[
            numeric_stage,
            temperature,
            humidity,
            quantity,
            sales_velocity,
            days_in_storage
        ]],
        columns=[
            "ripeness_stage",
            "temperature",
            "humidity",
            "quantity",
            "sales_velocity",
            "days_in_storage"
        ]
    )


    logger.info(
        "Spoilage model input: %s",
        features.to_dict(
            orient="records"
        )[0]
    )


    prediction = spoilage_model.predict(
        features
    )


    # Multi-output regression:
    #
    # [
    #   spoilage_probability,
    #   expected_waste_kg
    # ]

    result = np.asarray(
        prediction[0]
    ).reshape(-1)


    if result.size < 2:

        raise RuntimeError(
            "Unexpected spoilage model output. "
            f"Received: {prediction}"
        )


    spoilage_probability = float(
        result[0]
    )

    expected_waste = float(
        result[1]
    )


    # Keep probability inside valid range.

    spoilage_probability = float(
        np.clip(
            spoilage_probability,
            0.0,
            1.0
        )
    )


    # Expected waste should not be negative.

    expected_waste = max(
        0.0,
        expected_waste
    )


    logger.info(
        "Spoilage probability: %.4f | Expected waste: %.4f",
        spoilage_probability,
        expected_waste
    )


    return (
        numeric_stage,
        spoilage_probability,
        expected_waste
    )


# ============================================================
# LAYER 3
# PRICING MODEL
# ============================================================

def predict_discount(
    spoilage_probability: float,
    expected_waste: float,
    quantity: float,
    sales_velocity: float
):

    # Exact Layer-3 features.

    features = pd.DataFrame(
        [[
            spoilage_probability,
            expected_waste,
            quantity,
            sales_velocity
        ]],
        columns=[
            "spoilage_probability",
            "expected_waste_kg",
            "quantity",
            "sales_velocity"
        ]
    )


    logger.info(
        "Pricing model input: %s",
        features.to_dict(
            orient="records"
        )[0]
    )


    prediction = pricing_model.predict(
        features
    )


    discount = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )


    # Keep API output sensible.

    discount = float(
        np.clip(
            discount,
            0,
            100
        )
    )


    optimal_discount = int(
        round(discount)
    )


    logger.info(
        "Optimal discount: %s%%",
        optimal_discount
    )


    return optimal_discount


# ============================================================
# RETAILER RECOMMENDATION
# ============================================================

def get_retailer_action(
    stage_label: str,
    spoilage_probability: float,
    discount: int
):

    stage = stage_label.lower()


    if stage == "rotten":

        return (
            "Remove the product from sale and inventory. "
            "Product classified as rotten."
        )


    if spoilage_probability >= 0.80:

        return (
            f"Critical spoilage risk. Apply approximately "
            f"{discount}% discount and prioritize immediate sale."
        )


    if stage == "overripe":

        return (
            f"High deterioration risk. Apply approximately "
            f"{discount}% discount for faster inventory movement."
        )


    if stage == "ripe":

        return (
            f"Product is ready for sale. Recommended discount: "
            f"{discount}%."
        )


    return (
        "Product is currently relatively stable. "
        "Continue storage monitoring."
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "Smart Retail AI API is running",
        "pipeline": (
            "Ripeness → Spoilage → Pricing"
        )
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "models": {
            "ripeness": "loaded",
            "spoilage": "loaded",
            "pricing": "loaded"
        },

        "class_names": class_names,

        "image_size": IMG_SIZE,

        "preprocessing": (
            "RGB -> resize -> float32; "
            "no preprocess_input; no /255"
        )

    }


# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(

    file: UploadFile = File(...),

    temperature: float = Form(...),

    humidity: float = Form(...),

    quantity: float = Form(...),

    sales_velocity: float = Form(...),

    days_in_storage: int = Form(...)

):

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not np.isfinite(temperature):

        raise HTTPException(
            status_code=400,
            detail="Temperature must be a valid number."
        )


    if (
        not np.isfinite(humidity)
        or humidity < 0
        or humidity > 100
    ):

        raise HTTPException(
            status_code=400,
            detail="Humidity must be between 0 and 100."
        )


    if (
        not np.isfinite(quantity)
        or quantity <= 0
    ):

        raise HTTPException(
            status_code=400,
            detail="Quantity must be greater than 0."
        )


    if (
        not np.isfinite(sales_velocity)
        or sales_velocity < 0
    ):

        raise HTTPException(
            status_code=400,
            detail="Sales velocity cannot be negative."
        )


    if days_in_storage < 0:

        raise HTTPException(
            status_code=400,
            detail="Days in storage cannot be negative."
        )


    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        image_bytes = await file.read()


        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )


        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )


        image = image.convert(
            "RGB"
        )


    except HTTPException:

        raise


    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )


    except Exception:

        logger.exception(
            "Unable to read uploaded image."
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to process uploaded image."
        )


    # ========================================================
    # COMPLETE ML PIPELINE
    # ========================================================

    try:

        # ----------------------------------------------------
        # LAYER 1
        # MobileNetV2 Ripeness Classification
        # ----------------------------------------------------

        (
            stage_label,
            confidence,
            class_probabilities

        ) = predict_ripeness(
            image
        )


        # ----------------------------------------------------
        # LAYER 2
        # Spoilage + Waste Prediction
        # ----------------------------------------------------

        (
            numeric_stage,
            spoilage_probability,
            expected_waste

        ) = predict_spoilage(

            stage_label=stage_label,

            temperature=temperature,

            humidity=humidity,

            quantity=quantity,

            sales_velocity=sales_velocity,

            days_in_storage=days_in_storage

        )


        # ----------------------------------------------------
        # LAYER 3
        # Pricing Recommendation
        # ----------------------------------------------------

        optimal_discount = predict_discount(

            spoilage_probability=
                spoilage_probability,

            expected_waste=
                expected_waste,

            quantity=
                quantity,

            sales_velocity=
                sales_velocity

        )


        # ----------------------------------------------------
        # BUSINESS RECOMMENDATION
        # ----------------------------------------------------

        retailer_action = get_retailer_action(

            stage_label,

            spoilage_probability,

            optimal_discount

        )


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        return {

            "status":
                "Prediction Successful",

            "ripeness_stage":
                stage_label,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "ripeness_probabilities":
                class_probabilities,

            "numeric_ripeness_stage":
                numeric_stage,

            "spoilage_probability":
                round(
                    spoilage_probability,
                    4
                ),

            "expected_waste_kg":
                round(
                    expected_waste,
                    2
                ),

            "optimal_discount_percent":
                optimal_discount,

            "retailer_action":
                retailer_action

        }


    except Exception as e:

        logger.exception(
            "ML inference failed."
        )

        raise HTTPException(
            status_code=500,
            detail=f"ML inference failed: {str(e)}"
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )