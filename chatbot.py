import json 
import random 
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer

nltk.download('punkt')


# Cargar Y Entrenar

with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

stemmer = PorterStemmer
def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    return [stemmer.stem(word) for word in tokens]

x = []
y = []
tag_respuestas = {}

for intent in data['intents']:
    tag = intent['tag']
    tag_respuestas[tag] = intent['responses']
    for pattern in intent['patterns']:
        x.append(pattern)
        y.append(tag)

model = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize_and_stem)),
    ('clasifier', MultinomialNB())
])