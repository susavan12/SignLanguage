import json
import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

IMG_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")


def load_label_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def main():
    st.title("Sign Language Detection")

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_label_map(LABEL_MAP_PATH)

    image_file = st.camera_input("Capture sign image")

    if image_file is not None:
        image = Image.open(image_file)

        frame = np.array(image)

        processed = preprocess_frame(frame)

        prediction = model.predict(processed, verbose=0)

        predicted_index = np.argmax(prediction, axis=1)[0]
        label = labels[str(predicted_index)] if str(predicted_index) in labels else labels[predicted_index]

        confidence = float(np.max(prediction))

        st.image(image, caption="Captured Image")

        st.success(
            f"Prediction: {label}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )


if __name__ == "__main__":
    main()
