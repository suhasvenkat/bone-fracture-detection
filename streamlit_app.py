import streamlit as st
import cv2
import numpy as np
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
Upload an **X-ray image** and the model will:

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
    st.image(image, width='stretch')

    with st.spinner("Analyzing X-ray..."):

        label, confidence, heatmap = predict(image_np)

    st.subheader("Prediction Result")

    if label == "Fracture":
        st.error(f"⚠️ Fracture detected ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"✅ Normal bone ({confidence*100:.2f}% confidence)")

    st.subheader("Fracture Localization (Grad-CAM)")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original X-ray", width='stretch')

    with col2:
        st.image(heatmap, caption="Grad-CAM Heatmap", width='stretch')

st.markdown("---")

st.subheader("Project Information")

st.write(
"""
This deep learning system uses **Convolutional Neural Networks (CNNs)**  
to analyze medical X-ray images and detect bone fractures.

The model performs:

• Bone classification  
• Fracture detection  
• Visual explanation using **Grad-CAM**

Grad-CAM highlights the regions of the X-ray that most influenced the model's prediction.
"""
)