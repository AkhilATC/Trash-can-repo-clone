#!/bin/bash
set -e

NAMESPACE=kafka
CONNECTOR_VERSION=2.0.1
PLUGIN_DIR=plugins

# 1. Create namespace
kubectl create ns $NAMESPACE || true

# 2. Install Strimzi Operator (Kafka + Connect)
kubectl apply -n $NAMESPACE -f 'https://strimzi.io/install/latest?namespace=kafka'

# 3. Deploy Kafka cluster (CRD)
kubectl apply -n $NAMESPACE -f kafka-cluster.yaml

# 4. Build Kafka Connect image with Mongo plugin
cat <<EOF > Dockerfile
FROM quay.io/strimzi/kafka:latest-kafka-3.6.0
COPY $PLUGIN_DIR /opt/kafka/plugins/
EOF

docker build -t my-kafka-connect:latest .
kind load docker-image my-kafka-connect:latest

# 5. Deploy Kafka Connect using that image
kubectl apply -n $NAMESPACE -f kafka-connect.yaml

# 6. Create connectors (source + sink)
kubectl apply -n $NAMESPACE -f mongo-source.yaml
kubectl apply -n $NAMESPACE -f mongo-sink.yaml

