import requests
import json

KAFKA_CONNECT_URL = "http://localhost:8083/connectors"

# Connector configs
source_config = {
    "name": "mongo-source-A",
    "config": {
      "connector.class": "com.mongodb.kafka.connect.MongoSourceConnector",
      "tasks.max": "1",
      "connection.uri": "mongodb://atc:hype@172.17.0.1:27017/ATC?authSource=admin",
      "database": "ATC",
      "collection": "A",
      "publish.full.document.only": "true",
      "output.format.value": "json",
      "topic.prefix": "mongo"   # This will create topic: mongo.ATC.A
    }
}

# Sink connector: Kafka -> MongoDB
sink_config = {
    "name": "mongo-sink-B",
    "config": {
      "connector.class": "com.mongodb.kafka.connect.MongoSinkConnector",
      "tasks.max": "1",
      "topics": "mongo.ATC.A",  # must match the topic created by the source
      "connection.uri": "mongodb://atc:hype@172.17.0.1:27017/ATC?authSource=admin",
      "database": "ATC",
      "collection": "B",
      "key.converter": "org.apache.kafka.connect.storage.StringConverter",
      "value.converter": "org.apache.kafka.connect.json.JsonConverter",
      "value.converter.schemas.enable": "false"
    }
}


def create_connector(config):
    """POST a connector config to Kafka Connect"""
    resp = requests.post(
        KAFKA_CONNECT_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(config)
    )
    if resp.status_code in (200, 201):
        print(f"✅ Created connector: {config['name']}")
    elif resp.status_code == 409:
        print(f"⚠️ Connector {config['name']} already exists")
    else:
        print(f"❌ Failed to create {config['name']}: {resp.status_code} {resp.text}")


def list_connectors():
    """List all active connectors"""
    resp = requests.get(KAFKA_CONNECT_URL)
    if resp.ok:
        print("\n📋 Active connectors:")
        for c in resp.json():
            print(" -", c)
    else:
        print("❌ Failed to fetch connectors:", resp.status_code, resp.text)


if __name__ == "__main__":
    create_connector(source_config)
    create_connector(sink_config)
    list_connectors()

# connector status
#curl -s http://localhost:8083/connectors/mongo-sink-B/status | jq
# curl -s http://localhost:8083/connectors/mongo-source-A/status | jq

# curl -X DELETE http://localhost:8083/connectors/mongo-source
# curl -X DELETE http://localhost:8083/connectors/mongo-sink

"""

# Corrected configs

# Source connector: MongoDB -> Kafka
source_config = {
    "name": "mongo-source-A",
    "config": {
      "connector.class": "com.mongodb.kafka.connect.MongoSourceConnector",
      "tasks.max": "1",
      "connection.uri": "mongodb://atc:hype@172.17.0.1:27017/ATC?authSource=admin",
      "database": "ATC",
      "collection": "A",
      "publish.full.document.only": "true",
      "output.format.value": "json",
      "topic.prefix": "mongo"   # This will create topic: mongo.ATC.A
    }
}

# Sink connector: Kafka -> MongoDB
sink_config = {
    "name": "mongo-sink-B",
    "config": {
      "connector.class": "com.mongodb.kafka.connect.MongoSinkConnector",
      "tasks.max": "1",
      "topics": "mongo.ATC.A",  # must match the topic created by the source
      "connection.uri": "mongodb://atc:hype@172.17.0.1:27017/ATC?authSource=admin",
      "database": "ATC",
      "collection": "B",
      "key.converter": "org.apache.kafka.connect.storage.StringConverter",
      "value.converter": "org.apache.kafka.connect.json.JsonConverter",
      "value.converter.schemas.enable": "false"
    }
}









"""
