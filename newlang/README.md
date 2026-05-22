# Sign Language Detection System

This project provides a starter sign language detection system using Python and machine learning.
It includes scripts for collecting gesture images, training a CNN model, and running real-time detection using your webcam.

## Structure

- `collect_data.py`: Capture sign images from webcam and save them by label.
- `train_model.py`: Train a Keras CNN on the collected dataset.
- `app.py`: Load the trained model and perform live prediction from webcam.
- `gui.py`: Simple Tkinter interface for creating folders, collecting data, training, and running detection.
- `requirements.txt`: Python dependencies.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Collect Dataset

Create folders for each sign inside a `datasets` directory. For example:

```bash
mkdir datasets
mkdir datasets/A
mkdir datasets/B
mkdir datasets/C
```

Then collect images for a label:

```bash
python collect_data.py --label A --count 200
```

Repeat for each sign label.

## Train the Model

Run training after collecting data:

```bash
python train_model.py
```

For better performance, use data augmentation:

```bash
python train_model.py --augment --epochs 30
```

This saves:

- `sign_model.h5`
- `label_map.json`
- `training_history.png`

## Run Detection

Use the webcam-based real-time detector:

```bash
python app.py
```

Press `q` to quit.

## Optional GUI

A simple GUI is available to help with dataset setup, collection, training, and running detection:

```bash
python gui.py
```

Use the buttons to:
- create label folders under `datasets`
- collect sign images for a label
- train the model
- run the detector

## Notes

- Use good lighting and a plain background for best results.
- Collect multiple examples for each sign.
- You can adjust `IMG_SIZE`, `EPOCHS`, and model architecture in `train_model.py`.
