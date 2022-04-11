# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 02:25:32 2021

@author: chapter 3
"""

import pandas as pd
import numpy as np
from knowledge_function import build_chart
import ast
#Helper function to convert Nan to 0 and all other years to integers.
def convert_int(x):
    try:
        return int(x)
    except:
        return 0
   
#Load the data
    
df = pd.read_csv('movies_metadata.csv', low_memory=False)

#Print all the features (or columns) of the DataFrame
df.columns

#Only keep those features that we require 
df = df[['title','genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]

df.head()
#Convert release_date into pandas datetime format
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

#Extract year from the datetime
df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

#Apply convert_int to the year feature
df['year'] = df['year'].apply(convert_int)
#Drop the release_date column
df = df.drop('release_date', axis=1)

#Display the dataframe
df.head()
df.info()
#Print genres of the first movie
df.iloc[0]['genres']
#type(df['genres'])
#Import the literal_eval function from ast
from ast import literal_eval
#Convert all NaN into stringified empty lists
df['genres'] = df['genres'].fillna('[]')

#Apply literal_eval to convert stringified empty lists to the list object
df['genres'] = df['genres'].apply(literal_eval)
type(df['genres'])
#Convert list of dictionaries to a list of strings
df['genres'] = df['genres'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])
#print the genres of the first movie
df.iloc[0]['genres']
#print the genres of the fifth movie
df.iloc[4]['genres']
#df.head()
##
df[['runtime','vote_average','vote_count']]=df[['runtime','vote_average','vote_count']].fillna(0)
#df[["runtime","vote_average","vote_count"]].head()
##
##
#Create a new feature by exploding genres
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
type(s)
#Name the new feature as 'genre'
s.name = 'genre'

#Create a new dataframe gen_df which by dropping the old 'genres' feature and adding the new 'genre'.
gen_df = df.drop('genres', axis=1).join(s)

#Print the head of the new gen_df
gen_df.head(25)
### Ask for input
#Generate the chart for top animation movies and display top 5.
return_df=build_chart(gen_df)
print(return_df.head())
