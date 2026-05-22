import json
import os

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")


def open_camera(max_index=4):
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if cap.isOpened():
            print(f"Opened webcam on index {index}.")
            return cap
        cap.release()
    raise RuntimeError(
        "Could not open webcam. "
        "Close other camera apps, check camera permissions, or try a different camera index."
    )


def load_label_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train the model first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_label_map(LABEL_MAP_PATH)

    cap = open_camera()

    print("Starting real-time sign language detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        side = min(height, width)
        x = (width - side) // 2
        y = (height - side) // 2
        roi = frame[y:y + side, x:x + side]

        image = preprocess_frame(roi)
        prediction = model.predict(image, verbose=0)
        predicted_index = np.argmax(prediction, axis=1)[0]
        label = labels[predicted_index]
        confidence = float(np.max(prediction))

        cv2.rectangle(frame, (x, y), (x + side, y + side), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Sign Language Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
