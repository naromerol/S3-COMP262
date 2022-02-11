'''
COMP262 - Assignment 1
Nestor Romero - 301133331
Exercise 3
'''
#Basic imports
from copy import deepcopy
from email import header
import pandas as pd
import numpy as np
import random
from datetime import datetime

#scikit scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#Load positive and negative words
positive_df = pd.read_csv("positive-words.txt",sep="\t",encoding='latin1',header=None, names=["words"])
negative_df = pd.read_csv("negative-words.txt",sep="\t",encoding='latin1',header=None, names=["words"])
positive_words = set(list(positive_df["words"]))
negative_words = set(list(negative_df["words"]))

def calculate_sentiment_score(data):
    
    for index in data.index:
        
        tweet_words = set(data.iloc[index,1].split(" "))
        total_words = len(tweet_words)
        positives = tweet_words & positive_words
        negatives = tweet_words & negative_words
        
        pos_score = len(positives) / total_words
        neg_score = len(negatives) / total_words
        
        if pos_score == neg_score:
            sentiment_score = 'neutral'
        elif pos_score > neg_score:
            sentiment_score = 'positive'
        elif pos_score < neg_score:
            sentiment_score = 'negative'
        else:
            sentiment_score = 'neutral'
        
        
        data.loc[index,'positive_percentage'] = pos_score
        data.loc[index,'negative_percentage'] = neg_score
        data.loc[index,'predicted_sentiment_score'] = sentiment_score
        
    return data
        

### MAIN PROGRAM EXECUTION
#Load data into dataframe, analyze and remove user column
nestor_df = pd.read_csv('Artificial_Intelligence_data.csv', on_bad_lines='warn')
nestor_df = nestor_df.drop(columns=['user'], axis=1)

print('INITIAL DATA EXPLORATION')
print(nestor_df.head())
print(nestor_df.info())
print(nestor_df.describe())
print(nestor_df['sentiment'].value_counts())

#remove whitespaces
nestor_df['text'] = nestor_df['text'].apply(str.strip)

#calculate tweets lenght
nestor_df['tweet_len'] = nestor_df['text'].apply(len)

#calculate sentiment score
nestor_df = calculate_sentiment_score( nestor_df)
print(nestor_df['positive_percentage'].describe())
print(nestor_df['negative_percentage'].describe())

#calculate scores (accuracy / f1)
accuracy = accuracy_score(nestor_df['sentiment'], nestor_df['predicted_sentiment_score'])
print(f'Accuracy Score: {accuracy}')

f1_macro = f1_score(nestor_df['sentiment'], nestor_df['predicted_sentiment_score'], average='macro')
print(f'F1-Macro Score: {f1_macro}')
f1_weighted = f1_score(nestor_df['sentiment'], nestor_df['predicted_sentiment_score'], average='weighted')
print(f'F1-Weighted Score: {f1_weighted}')
f1_micro = f1_score(nestor_df['sentiment'], nestor_df['predicted_sentiment_score'], average='micro')
print(f'F1-Micro Score: {f1_micro}')