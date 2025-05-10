# Distributed Machine Learning with Kafka and TensorFlow

This project demonstrates a distributed machine learning system using Kafka as a message broker and TensorFlow for model training. The system processes the CIFAR-10 dataset in a distributed manner.

## Prerequisites

- Python 3.8+
- Apache Kafka
- Java 8+ (required for Kafka)

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Start Kafka server:

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka server
bin/kafka-server-start.sh config/server.properties
```

3. Create Kafka topics:

```bash
bin/kafka-topics.sh --create --topic cifar10_training --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
bin/kafka-topics.sh --create --topic cifar10_testing --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

## Usage

1. Start the consumer (model trainer) in one terminal:

```bash
python consumer.py
```

2. Start the producer (data sender) in another terminal:

```bash
python producer.py
```

## System Architecture

- **Producer (producer.py)**:

  - Loads CIFAR-10 dataset
  - Sends images and labels to Kafka topics
  - Separates training and testing data

- **Consumer (consumer.py)**:
  - Receives data from Kafka topics
  - Implements a CNN model using TensorFlow
  - Trains the model on batches of data
  - Evaluates model performance on test data

## Model Architecture

The model is a simple CNN with the following layers:

- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- 2 Dense layers
- Output layer with 10 classes (CIFAR-10)

## Notes

- The system uses batch processing to train the model
- Training and testing are performed in separate threads
- The model is trained incrementally as data arrives
- Test accuracy is reported periodically
