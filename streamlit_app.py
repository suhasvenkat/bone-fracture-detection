import streamlit as st
import numpy as np
import cv2
from PIL import Image
from predictions import predict

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Bone Fracture Detection System")

st.write(
"""
Upload an **X-ray image** and the AI model will:

1️⃣ Identify the **bone type**

2️⃣ Detect whether the bone is **fractured or normal**
"""
)

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Uploaded X-ray")
    st.image(image, width="stretch")

    with st.spinner("Analyzing X-ray..."):

        body_part, label, confidence = predict(image_np)

    st.subheader("Prediction Result")

    st.write(f"**Detected Bone:** {body_part}")

    if label == "Fracture":
        st.error(f"⚠️ Fracture detected ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"✅ Normal bone ({confidence*100:.2f}% confidence)")


st.markdown("---")

st.subheader("Project Information")

st.write(
"""
This deep learning system uses **ResNet50 convolutional neural networks** to analyze medical X-ray images.

The pipeline works in two stages:

1️⃣ **Body part classification**
- Elbow
- Hand
- Shoulder

2️⃣ **Fracture detection**
- A specialized model for each bone type

This approach improves accuracy by allowing the fracture model to specialize in a specific anatomical region.
"""
)