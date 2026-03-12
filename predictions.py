import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# load trained model
model = load_model("weights/fracture_model.h5")

IMG_SIZE = 224


def preprocess_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# Grad-CAM
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(heatmap, original_image, alpha=0.4):

    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * alpha + original_image

    return superimposed.astype("uint8")


# -----------------------------
# Prediction function
# -----------------------------
def predict(image):

    original_img = image.copy()

    img_array = preprocess_image(image)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Fracture"
        confidence = prediction
    else:
        label = "Normal"
        confidence = 1 - prediction

    # generate GradCAM
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name="conv2d_3"   # change if your last conv layer name is different
    )

    heatmap_image = overlay_heatmap(heatmap, original_img)

    return label, confidence, heatmap_image