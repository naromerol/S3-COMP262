# COMP262 - MIDTERM
# NESTOR ROMERO - 301133331

text = "Wherever you are from, and wherever you would like to be, SaGE would like to help you broaden your horizons, build your global network, and achieve academic, personal, and professional success during your stay at Centennial"
# print(text)

#YOUR CODE GOES HERE
# 1. IMPORT LIBRARIES
import spacy

def get_nouns (text):
    nlp=spacy.load("en_core_web_sm")
    doc=nlp(text) # Coverting the text into a spacy Doc  
    # YOUR CODE GOES HERE
    print('WORD\tPOS TAG')
    for token in doc:
        if(token.pos_ == 'NOUN' or token.pos_ == 'PROPN'):
            print(token, '\t', token.pos_)
        
get_nouns(text)