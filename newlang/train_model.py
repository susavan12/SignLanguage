import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_DATA_DIR = 'datasets'
DEFAULT_MODEL_PATH = 'sign_model.h5'
DEFAULT_LABEL_MAP_PATH = 'label_map.json'


def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_generators(data_dir, batch_size, augment):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2,
            rotation_range=12,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            zoom_range=0.12,
            brightness_range=(0.7, 1.3),
            horizontal_flip=True,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=batch_size,
        subset='training',
        shuffle=True,
    )

    validation_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=batch_size,
        subset='validation',
        shuffle=False,
    )

    return train_generator, validation_generator


def plot_history(history, output_path='training_history.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved training history plot to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description='Train the sign language recognition model.')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Path to the dataset directory.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help='Output model file path.')
    parser.add_argument('--label-map-path', type=str, default=DEFAULT_LABEL_MAP_PATH, help='Output label map file path.')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation during training.')
    args = parser.parse_args()

    train_generator, validation_generator = create_generators(args.data_dir, args.batch_size, args.augment)
    label_names = [None] * len(train_generator.class_indices)
    for label_name, index in train_generator.class_indices.items():
        label_names[index] = label_name

    model = build_model(len(label_names))
    model.summary()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
    )

    model.save(args.model_path)
    print(f"Saved trained model to {args.model_path}.")

    with open(args.label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_names, f, indent=2)
    print(f"Saved label map to {args.label_map_path}.")

    plot_history(history)


if __name__ == '__main__':
    main()
