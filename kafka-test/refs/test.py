from pymongo import MongoClient
import time

# Connect to your Mongo
client = MongoClient("mongodb://atc:hype@localhost:27017/ATC?authSource=admin")
db = client["ATC"]

# Insert a test doc into A
doc = {"test": "hello from Python", "ts": time.time()}
result = db.A.insert_one(doc)
print("Inserted into A:", result.inserted_id)

# Wait for Kafka + Sink to push into B
time.sleep(5)

# Check B
found = db.B.find_one({"_id": result.inserted_id})
print("Found in B:", found)
