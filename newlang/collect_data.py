import argparse
import os
import cv2

IMG_SIZE = 64


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main(label: str, count: int, output_dir: str):
    label_dir = os.path.join(output_dir, label)
    ensure_dir(label_dir)

    cap = open_camera()

    saved = len([name for name in os.listdir(label_dir) if name.endswith('.png')])
    print(f"Collecting images for label '{label}' into {label_dir}")
    print("Press 'q' to stop early.")

    while saved < count:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        side = min(height, width)
        x = (width - side) // 2
        y = (height - side) // 2
        roi = frame[y:y + side, x:x + side]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        filename = os.path.join(label_dir, f"{saved + 1:04d}.png")
        cv2.imwrite(filename, gray)
        saved += 1

        cv2.rectangle(frame, (x, y), (x + side, y + side), (0, 255, 0), 2)
        cv2.putText(frame, f"Label: {label} ({saved}/{count})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Sign Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images for label '{label}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect sign language image data from webcam.")
    parser.add_argument('--label', type=str, required=True, help='Label name for the sign class.')
    parser.add_argument('--count', type=int, default=200, help='Number of images to collect.')
    parser.add_argument('--output-dir', type=str, default='datasets', help='Path to store captured images.')
    args = parser.parse_args()

    main(args.label, args.count, args.output_dir)
