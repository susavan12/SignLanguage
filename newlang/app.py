import json
import os

import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

IMG_SIZE = 64

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")


def load_label_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Resize image
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img.astype("float32") / 255.0

    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def main():
    st.title("🤟 Sign Language Detection")

    # Check model existence
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        return

    # Load model and labels
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_label_map(LABEL_MAP_PATH)

    # Input selection
    input_method = st.radio(
        "Choose input method",
        ["Camera", "Upload Image"]
    )

    image = None

    # Camera option
    if input_method == "Camera":
        image_file = st.camera_input(
            "Capture sign image"
        )

        if image_file is not None:
            image = Image.open(image_file)

    # Upload option
    else:
        uploaded_file = st.file_uploader(
            "Upload sign image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    # Prediction
    if image is not None:

        frame = np.array(image)

        processed = preprocess_frame(frame)

        prediction = model.predict(
            processed,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction,
            axis=1
        )[0]

        # Handle both string and integer keys
        if str(predicted_index) in labels:
            label = labels[str(predicted_index)]
        else:
            label = labels[predicted_index]

        confidence = float(
            np.max(prediction)
        )

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

        st.success(
            f"Prediction: {label}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )


if __name__ == "__main__":
    main()
