import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np

# Page config
st.set_page_config(
    page_title="Emotion Recognition",
    layout="centered"
)

# Title
st.title("😃 Emotion Recognition App")
st.write("Upload a face image and detect emotion")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show image in browser UI
    st.image(
        img_rgb,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Emotion detection
    with st.spinner("Detecting emotion..."):
        result = DeepFace.analyze(
            img_path=img_rgb,
            actions=["emotion"],
            enforce_detection=False
        )

    emotion = result[0]["dominant_emotion"]

    # Show result
    st.success(f"Detected Emotion: **{emotion.upper()}**")
