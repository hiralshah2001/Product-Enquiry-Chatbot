import streamlit as st
import json
import random
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Load resources
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Functions for preprocessing and predictions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['intent'] == tag:
            return random.choice(i['responses'])

# Streamlit app
st.title("AI Chatbot")
st.write("Welcome! Ask me anything!")

# Input from the user
user_input = st.text_input("You:", "Type your message here...")

if st.button("Send"):
    if user_input.strip() != "":
        intents_predicted = predict_class(user_input)
        response = get_response(intents_predicted, intents)
        st.text_area("Chatbot:", response, height=150, max_chars=None)
    else:
        st.warning("Please enter a message.")

st.sidebar.title("About the Chatbot")
st.sidebar.info("This chatbot is built using a trained neural network to respond to user queries.")
