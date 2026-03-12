import streamlit as st
from PIL import Image
import tempfile
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

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Run Prediction"):

        with st.spinner("Analyzing X-ray..."):

            try:

                # save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    image_path = tmp.name

                # STEP 1: Detect bone type
                bone_type = predict(image_path, "Parts")

                if bone_type.startswith("Not identified"):
                    st.error("❌ Bone type could not be identified.")
                else:

                    # STEP 2: Detect fracture
                    fracture_result = predict(image_path, bone_type)

                    st.subheader("Prediction Result")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"Bone Type: **{bone_type}**")

                    with col2:

                        if fracture_result == "fractured":
                            st.error("⚠️ Fracture Detected")
                        else:
                            st.success("✅ No Fracture Detected")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")


st.markdown("---")

st.markdown(
"""
### Project Information

This deep learning system:

• Classifies **bone type (Elbow, Hand, Shoulder, Wrist)**  
• Detects **fractures using CNN models**  
• Built using **TensorFlow / Keras**

"""
)