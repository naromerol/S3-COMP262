'''
COMP262 - Assignment 1
Nestor Romero - 301133331
Exercise 2
'''
#Basic imports
from copy import deepcopy
import pandas as pd
import numpy as np
import random
from datetime import datetime

#nlp imports
import string, re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, KeyedVectors



def preprocess_tweet(t):
    '''
    Function to execute common preprocessing steps over each tweet
    Each tweet is received as a row from a dataframe
    [sentiment, text]
    Performs:
    - Lowercasing
    - Removes punctuation, digits, whitespace
    - Tokenization
    - Remove stop words 
    '''
    
    #lowercase tweet text
    t['text'] = t['text'].lower()
    
    #remove punctuation, digits and whitespace
    t['text'] = t['text'].translate(str.maketrans('','', string.punctuation))
    t['text'] = re.sub(r'\d+','', t['text'])
    t['text'] = t['text'].strip()
    
    #tokenize string and remove stop words
    t_tokenized = word_tokenize(t['text'])
    t_tokenized_nostop = [i for i in t_tokenized if not i in stop_words]
    t['text'] = ' '.join(t_tokenized_nostop) 

def prepare_word2vec():
    '''
    Utilitary method to prepare word2vec model 
    '''
    #PATH to the GoogleNews Negative bin file
    bin_file_path='GoogleNews-vectors-negative300.bin'
    word2vec_model = KeyedVectors.load_word2vec_format(bin_file_path, binary=True)
    return word2vec_model
    
    
def augment_tweets_df(dataframe, word2vec_model):
    '''
    Function to execute word augmentation tasks
    Prtoduces a new dataframe with random insert of words
    '''
    aux_df = dataframe.copy(deep=True)
    
    for index in aux_df.index:
        t = aux_df.iloc[index]
        t_tokenized = word_tokenize(t['text'])
        word1_found, word2_found = False, False
        
        while word1_found == False and word2_found == False:
            try:
                rand_pos1 = random.randint(0, len(t_tokenized)-1)
                rand_pos2 = random.randint(0, len(t_tokenized)-1)
                word1 = t_tokenized[rand_pos1]
                synonims1 = word2vec_model.most_similar(word1)
                word2 = t_tokenized[rand_pos2]
                synonims2 = word2vec_model.most_similar(word2)
                
                word1_found, word2_found = True, True
                
                if word1_found and word2_found:
                    print(f'{word1} {synonims1[0]}')
                    print(f'{word2} {synonims2[0]}')
                    replace1 = synonims1[0][0]
                    replace2 = synonims2[0][0]
                    t_tokenized[rand_pos1] = replace1
                    t_tokenized[rand_pos2] = replace2
                    print(aux_df.iloc[index]['text'])
                    aux_df.iloc[index]['text'] = ' '.join(t_tokenized)
                
            except KeyError as ke:
                print(ke)
    return aux_df

#LIBRARY AND MODEL SETUP
print(datetime.now().time())
print('Loading stop words')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print('Loading word2vec')
word2vec_model = prepare_word2vec()
print(datetime.now().time())
    
### MAIN PROGRAM EXECUTION
#Load data into dataframe, analyze and remove user column
nestor_df = pd.read_csv('Artificial_Intelligence_mini.csv')
nestor_df = nestor_df.drop(columns=['user'], axis=1)
# print(nestor_df.head())

#Preprocess data for analysis
nestor_df.apply(preprocess_tweet, axis=1)
raw_tweet_example = nestor_df.loc[0,'text']
preprocessed_tweet_example = nestor_df.loc[0,'text']

#Word2Vec augmentation - random insertion
nestor_df_aux = augment_tweets_df(nestor_df, word2vec_model)
nestor_df_after_word_augmenter = pd.concat([nestor_df,nestor_df_aux], axis=0, ignore_index=True)
nestor_df_after_word_augmenter.to_csv('nestor_df_after_random_insertion.csv')

print(datetime.now().time())