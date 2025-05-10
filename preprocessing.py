import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def apply_data_augmentation(image):
    if np.random.random() > 0.5:
        image = np.fliplr(image)

    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        image = tf.image.rot90(image, k=int(angle / 90))

    if np.random.random() > 0.5:
        image = tf.image.random_brightness(image, 0.2)

    return image


def preprocess_image(image):
    pil_image = Image.fromarray(image)

    if pil_image.size != (32, 32):
        pil_image = pil_image.resize((32, 32), Image.LANCZOS)

    image = np.array(pil_image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = normalize_image(image)

    return image


def process_batch(images, labels, augment=False):
    processed_images = []
    processed_labels = []

    for image, label in zip(images, labels):
        processed_image = preprocess_image(image)

        if augment:
            processed_image = apply_data_augmentation(processed_image)

        processed_images.append(processed_image)
        processed_labels.append(label)

    return np.array(processed_images), np.array(processed_labels)
