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

print("Loading models from:", BASE_DIR)

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


# -----------------------------
# GradCAM
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img):

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return superimposed


# -----------------------------
# Prediction Pipeline
# -----------------------------
def predict(image):

    original_img = image.copy()

    img = preprocess(image)

    # Step 1: detect body part
    body_pred = body_model.predict(img)
    body_class = np.argmax(body_pred)

    body_parts = ["Elbow", "Hand", "Shoulder"]
    body_part = body_parts[body_class]

    # Step 2: select fracture model
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
        confidence = float(fracture_prob)
    else:
        label = "Normal"
        confidence = float(1 - fracture_prob)

    # Step 4: GradCAM
    heatmap = make_gradcam_heatmap(img, model)
    heatmap_img = overlay_heatmap(heatmap, original_img)

    return body_part, label, confidence, heatmap_img