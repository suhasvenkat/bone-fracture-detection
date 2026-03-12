import os
import cv2
import numpy as np
import tensorflow as tf
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


# -----------------------------
# GradCAM
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(heatmap, image, alpha=0.4):

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * alpha + image

    return superimposed.astype("uint8")


# -----------------------------
# Prediction pipeline
# -----------------------------
def predict(image):

    original_img = image.copy()

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

    # Step 4: GradCAM
    heatmap = make_gradcam_heatmap(img, model, "conv5_block3_out")

    heatmap_img = overlay_heatmap(heatmap, original_img)

    return body_part, label, confidence, heatmap_img