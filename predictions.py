import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(__file__)

# Load models
body_model = load_model(os.path.join(BASE_DIR, "weights/ResNet50_BodyParts.h5"))
elbow_model = load_model(os.path.join(BASE_DIR, "weights/ResNet50_Elbow_frac.h5"))
hand_model = load_model(os.path.join(BASE_DIR, "weights/ResNet50_Hand_frac.h5"))
shoulder_model = load_model(os.path.join(BASE_DIR, "weights/ResNet50_Shoulder_frac.h5"))

IMG_SIZE = 224


def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(image):

    img = preprocess(image)

    # Step 1: detect body part
    body_pred = body_model.predict(img)
    body_class = np.argmax(body_pred)

    body_parts = ["Elbow", "Hand", "Shoulder"]
    body_part = body_parts[body_class]

    # Step 2: choose correct fracture model
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