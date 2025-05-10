import tensorflow as tf
from kafka import KafkaProducer
import json
import numpy as np
from tqdm import tqdm
from preprocessing import process_batch


def create_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    )


def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def send_data_to_kafka(
    producer, x_data, y_data, topic_name, augment=False, batch_size=100
):
    """
    Gửi dữ liệu đã được tiền xử lý qua Kafka theo batch
    Args:
        producer: Kafka producer
        x_data: Dữ liệu ảnh
        y_data: Nhãn
        topic_name: Tên topic Kafka
        augment: Có áp dụng data augmentation hay không
        batch_size: Kích thước batch để xử lý
    """
    total_samples = len(x_data)

    for start_idx in tqdm(range(0, total_samples, batch_size)):
        end_idx = min(start_idx + batch_size, total_samples)

        # Lấy batch dữ liệu
        batch_x = x_data[start_idx:end_idx]
        batch_y = y_data[start_idx:end_idx]

        # Tiền xử lý batch
        processed_images, processed_labels = process_batch(batch_x, batch_y, augment)

        # Gửi từng mẫu trong batch
        for i, (image, label) in enumerate(zip(processed_images, processed_labels)):
            message = {
                "image": image.tolist(),
                "label": int(label[0]),
                "index": start_idx + i,
                "preprocessed": True,
                "batch_index": i,
            }

            producer.send(topic_name, value=message)

        # Flush sau mỗi batch
        producer.flush()


def main():
    # Khởi tạo producer
    producer = create_kafka_producer()

    try:
        # Load CIFAR-10 data
        print("Loading CIFAR-10 data...")
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()

        # Gửi dữ liệu training với augmentation
        print("Sending training data...")
        send_data_to_kafka(
            producer, x_train, y_train, "cifar10_training", augment=True, batch_size=100
        )

        # Gửi dữ liệu testing không có augmentation
        print("Sending test data...")
        send_data_to_kafka(
            producer, x_test, y_test, "cifar10_testing", augment=False, batch_size=100
        )

        print("Data sending completed!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Đóng producer
        producer.close()


if __name__ == "__main__":
    main()
