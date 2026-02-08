import random

EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Fear"]

def predict_emotion(features):
    return random.choice(EMOTIONS)