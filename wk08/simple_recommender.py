# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 07:54:35 2021

@author: simple recommender -chapter 2
"""

import pandas as pd
import numpy as np


df = pd.read_csv('movies_metadata.csv', low_memory=False)
df.head()
#Calculate the number of votes garnered by the 80th percentile movie
m = df['vote_count'].quantile(0.99)
print(f'Votes quantile: {m}')

#Only consider movies longer than 45 minutes and shorter than 300 minutes
q_movies = df[(df['runtime'] >= 45) & (df['runtime'] <= 300)]
# q_movies.info()

#Only consider movies that have garnered more than m votes
q_movies = q_movies[q_movies['vote_count'] >= m]

#Inspect the number of movies that made the cut
print(q_movies.shape)

# Calculate C
C = df['vote_average'].mean()
C
# Function to compute the IMDB weighted rating for each movie
def weighted_rating(x, m=m, C=C):
    
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)

# Compute the score using the weighted_rating function defined above
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies in descending order of their scores
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 25 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score', 'runtime']].head(25))