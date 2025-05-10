from kafka import KafkaConsumer
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import threading
from queue import Queue
import time


class ModelTrainer:
    def __init__(self):
        self.model = self.create_model()
        self.training_queue = Queue()
        self.testing_queue = Queue()
        self.batch_size = 32
        self.training_data = []
        self.training_labels = []
        self.testing_data = []
        self.testing_labels = []

    def create_model(self):
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(10),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def process_training_data(self):
        while True:
            if len(self.training_data) >= self.batch_size:
                batch_data = np.array(self.training_data[: self.batch_size])
                batch_labels = np.array(self.training_labels[: self.batch_size])

                # Train the model
                self.model.train_on_batch(batch_data, batch_labels)

                # Remove processed data
                self.training_data = self.training_data[self.batch_size :]
                self.training_labels = self.training_labels[self.batch_size :]

                print(
                    f"Trained on batch. Current training data size: {len(self.training_data)}"
                )
            time.sleep(0.1)

    def process_testing_data(self):
        while True:
            if len(self.testing_data) >= self.batch_size:
                batch_data = np.array(self.testing_data[: self.batch_size])
                batch_labels = np.array(self.testing_labels[: self.batch_size])

                # Evaluate the model
                test_loss, test_acc = self.model.evaluate(
                    batch_data, batch_labels, verbose=0
                )
                print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

                # Remove processed data
                self.testing_data = self.testing_data[self.batch_size :]
                self.testing_labels = self.testing_labels[self.batch_size :]
            time.sleep(0.1)


def main():
    # Create Kafka consumer
    consumer = KafkaConsumer(
        bootstrap_servers=["localhost:9092"],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="cifar10_consumer_group",
    )

    # Subscribe to topics
    consumer.subscribe(["cifar10_training", "cifar10_testing"])

    # Create model trainer
    trainer = ModelTrainer()

    # Start processing threads
    training_thread = threading.Thread(
        target=trainer.process_training_data, daemon=True
    )
    testing_thread = threading.Thread(target=trainer.process_testing_data, daemon=True)
    training_thread.start()
    testing_thread.start()

    print("Starting to consume messages...")

    try:
        for message in consumer:
            data = message.value
            image = np.array(data["image"])
            label = data["label"]

            if message.topic == "cifar10_training":
                trainer.training_data.append(image)
                trainer.training_labels.append(label)
            else:  # cifar10_testing
                trainer.testing_data.append(image)
                trainer.testing_labels.append(label)

    except KeyboardInterrupt:
        print("Stopping consumer...")
        consumer.close()


if __name__ == "__main__":
    main()
