
intents = {
    "greeting": 0,
    "order_food": 1,
    "weather": 2
}

data = [
    # Greeting: 10 samples
    ("Hello", 0),
    ("Hi there", 0),
    ("Good morning", 0),
    ("Hey", 0),
    ("Hello, how are you?", 0),
    ("Good evening", 0),
    ("Hi, nice to meet you", 0),
    ("Hey there", 0),
    ("Howdy", 0),
    ("Greetings", 0),

    # Order food: 10 samples
    ("I want to order pizza", 1),
    ("Can I get a burger?", 1),
    ("I would like some food", 1),
    ("Order a chicken sandwich", 1),
    ("Get me a pizza", 1),
    ("I want to buy a meal", 1),
    ("Can you order noodles?", 1),
    ("I would like a cheeseburger", 1),
    ("Please order some pasta", 1),
    ("I want to eat pizza tonight", 1),

    # Weather: 10 samples
    ("What is the weather today?", 2),
    ("Will it rain today?", 2),
    ("Is it sunny outside?", 2),
    ("Tell me the temperature", 2),
    ("Is it going to rain tomorrow?", 2),
    ("How hot is it today?", 2),
    ("Give me the weather forecast", 2),
    ("Is it cloudy outside?", 2),
    ("Will tomorrow be sunny?", 2),
    ("What is the temperature outside?", 2),
]

texts = [text for text, label in data]
labels = [label for text, label in data]

print("Total samples:", len(data))       # 30
print("Number of intents:", len(intents)) # 3
