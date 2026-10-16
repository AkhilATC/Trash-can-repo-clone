"""
Step 1a: A tiny hand-written dataset.

Each example is (sentence, label_index). The model will learn to map
a sentence to one of the emotion emojis below.
"""

# The 6 classes our model can predict. The index in this list IS the label.
EMOJIS = ["😊", "😢", "😠", "😨", "😍", "😲"]
EMOTION_NAMES = ["joy", "sadness", "anger", "fear", "love", "surprise"]

# Small on purpose: easy to read, trains in seconds on CPU.
_RAW = {
    "joy": [
        "i just got the job",
        "today was a great day",
        "we won the match",
        "i passed my exam",
        "this is the best news ever",
        "i am so happy right now",
        "finally the weekend is here",
        "my friends threw me a party",
        "i feel amazing today",
        "the sun is shining and i feel good",
        "we are going on a holiday",
        "that joke made me laugh so hard",
    ],
    "sadness": [
        "my dog passed away",
        "i miss my family",
        "i failed the exam",
        "nobody came to my party",
        "i feel so lonely tonight",
        "she left and never came back",
        "i lost my job today",
        "everything feels empty",
        "i cried all night",
        "my best friend moved away",
        "i feel so down today",
        "the movie ending made me cry",
    ],
    "anger": [
        "i hate waiting in this line",
        "he lied to me again",
        "stop shouting at me",
        "this traffic is driving me crazy",
        "they broke my phone",
        "i am so angry right now",
        "why do you never listen",
        "the service here is terrible",
        "someone stole my bike",
        "i am sick of this nonsense",
        "he cheated in the game",
        "that is so unfair",
    ],
    "fear": [
        "i heard a noise in the dark",
        "i am scared of the exam results",
        "someone is following me",
        "the storm is getting worse",
        "i am afraid to fly",
        "there is a spider on my bed",
        "i am nervous about the surgery",
        "what if something bad happens",
        "the house is creepy at night",
        "i am terrified of heights",
        "the lights went out suddenly and i panicked",
        "i am worried about my future",
    ],
    "love": [
        "i love you so much",
        "you mean the world to me",
        "i adore my little sister",
        "my heart is full when i am with you",
        "i love my family",
        "you are my favourite person",
        "i miss your hugs my love",
        "you make me feel special",
        "i love this song so much",
        "my cat is the cutest i love her",
        "we are getting married",
        "you are the love of my life",
    ],
    "surprise": [
        "wow i did not expect that",
        "no way that really happened",
        "i can not believe it",
        "what a shocking twist",
        "they threw me a surprise party",
        "oh my god is that real",
        "i never saw that coming",
        "wait what just happened",
        "that came out of nowhere",
        "unbelievable he actually did it",
        "suddenly the whole room went quiet",
        "is this for real",
    ],
}


def get_dataset():
    """Return a list of (sentence, label_index) pairs."""
    data = []
    for name, sentences in _RAW.items():
        label = EMOTION_NAMES.index(name)
        for s in sentences:
            data.append((s, label))
    return data


if __name__ == "__main__":
    data = get_dataset()
    print(f"{len(data)} examples, {len(EMOJIS)} classes")
    for sentence, label in data[::12]:  # one example per class
        print(f"  {EMOJIS[label]}  {EMOTION_NAMES[label]:<8} <- {sentence!r}")
