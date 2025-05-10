# Lab 4 DS200

This is my repo for lab 4 of course DS2000 Big data for my study in UIT

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
  - Do some basic preprocessing
  - Sends images and labels to Kafka topics
  - Separates training and testing data

- **Consumer (consumer.py)**:
  - Receives data from Kafka topics
  - Implements a CNN model using TensorFlow
  - Trains the model on batches of data
  - Evaluates model performance on test data
