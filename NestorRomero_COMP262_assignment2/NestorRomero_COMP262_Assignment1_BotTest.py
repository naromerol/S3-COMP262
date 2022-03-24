######
# COMP262 - NLP - ASSIGNMENT 2
# NESTOR ROMERO - 301133331
######

import pickle
import numpy as np
import os
import json
import re
import random
import keras.models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorboard import summary


# LOAD MODEL OBJECTS FROM DISK
chatbot_intent_model = keras.models.load_model(
    './complaint_intent_model.model')
# print(chatbot_intent_model.summary())
tokenizer = pickle.load(open('./tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('./label_encoder.pkl', 'rb'))

# LOAD DATASET FOR FINAL RESPONSE
dataset = json.load(open('nestor_intents.json', encoding='utf-8'))
intents = []  # holds the original list of intents >> len( number_of_intents )
# holds the original responses for each intent >> len( number_of_intents )
responses = []

for record in dataset['intents']:
    intent = record['tag']
    intents.append(intent)
    responses.append(record['responses'])


# AUXILIARY FUNCTIONS
def clean_input(input):
    # Strip and lowercase user input
    input = input.strip().lower()
    return input


def preprocess_input(input):
    input = re.sub(r'[^\w\s]', '', str.lower(input.strip()))
    vocabulary_size = len(tokenizer.word_index)
    # Create vector from input
    vectors = tokenizer.texts_to_sequences([input])
    
    # Create sequence from vector
    sequence = pad_sequences(vectors, maxlen=vocabulary_size)
    return sequence


def select_answer(intent_code):
    # Select answer text based on the class id from label encoder!!!
    intent = ''
    response = ''
    intent_code = int(intent_code)
    intent_label = label_encoder.inverse_transform([intent_code])[0]
    
    try:
        intent_index = intents.index(intent_label)
        resp_id = random.randint(0, len(responses[intent_index])-1)
        intent = intents[intent_index]
        response = responses[intent_index][resp_id]
    except ValueError as ve:
        intent = '-Not found-'
        response = '-Not found-'
        print(ve)
    
    return intent, response


# MAIN PROGRAM EXECUTION
user_input = ''
last_response = ''
conversation = ''
debug_mode = False

# Clear console for better readability
os.system('cls')

while True:
    
    if last_response == '':
        print('Welcome, let us start our conversation')
        conversation += 'Welcome, let us start our conversation'

    user_input = input('(Type "Exit" to finish) << : ')
    user_input = clean_input(user_input)
    

    # Exit condition
    if user_input == 'exit':
        break

    # Record conversation
    conversation += '\n'
    conversation += f'[User] {user_input}'    

    # Prediction
    sequence = preprocess_input(user_input)
    prediction = chatbot_intent_model.predict(sequence)
    pred_label = prediction.argmax(axis=1)
    intent, response = select_answer(pred_label)

    if debug_mode:
        print(f'\nReceived text: {user_input}')
        print(f'\n\nSequence: {sequence}')
        print(f'\n\nModel Prediction: {prediction}')
        print(f'\n\nPredicted Label: {pred_label}')
        
    
    last_response = f'[Chatbot] [{intent}] {response}'
    print(last_response)

    conversation += '\n'
    conversation += last_response
    conversation += '\n'
    
    #input('Press any key to continue our conversation...')

print('BYE BYE')
print('#' * 50)
print('SUMMARY: ')
print('#' * 50)
print(conversation)
print('#' * 50)