import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
BODY_MODEL_PATH = os.path.join(BASE_DIR, "weights", "ResNet50_BodyParts.h5")
ELBOW_MODEL_PATH = os.path.join(BASE_DIR, "weights", "ResNet50_Elbow_frac.h5")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "weights", "ResNet50_Hand_frac.h5")
SHOULDER_MODEL_PATH = os.path.join(BASE_DIR, "weights", "ResNet50_Shoulder_frac.h5")

# Debug (helps Streamlit logs)
print("Loading models from:")
print(BODY_MODEL_PATH)

# Check files exist
if not os.path.exists(BODY_MODEL_PATH):
    raise FileNotFoundError(f"Missing model file: {BODY_MODEL_PATH}")

# Load models
body_model = load_model(BODY_MODEL_PATH)
elbow_model = load_model(ELBOW_MODEL_PATH)
hand_model = load_model(HAND_MODEL_PATH)
shoulder_model = load_model(SHOULDER_MODEL_PATH)

IMG_SIZE = 224


def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(image):

    img = preprocess(image)

    # Step 1: body part classification
    body_pred = body_model.predict(img)
    body_class = np.argmax(body_pred)

    body_parts = ["Elbow", "Hand", "Shoulder"]
    body_part = body_parts[body_class]

    # Step 2: choose fracture model
    if body_part == "Elbow":
        model = elbow_model
    elif body_part == "Hand":
        model = hand_model
    else:
        model = shoulder_model

    # Step 3: fracture prediction
    fracture_prob = model.predict(img)[0][0]

    if fracture_prob > 0.5:
        label = "Fracture"
        confidence = fracture_prob
    else:
        label = "Normal"
        confidence = 1 - fracture_prob

    return body_part, label, confidence