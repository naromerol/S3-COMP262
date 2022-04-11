# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:10:00 2021

@author: mhabayeb
"""
import pandas as pd
import numpy as np
#Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('wk11/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

users.head()
#Load the u.item file into a dataframe
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('wk11/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

movies.head()

#Remove all information except Movie ID and title
movies = movies[['movie_id', 'title']]

#Load the u.data file into a dataframe
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv('wk11/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

ratings.head()

#Drop the timestamp column
ratings = ratings.drop('timestamp', axis=1)
###### Model based
#Import the required classes and methods from the surprise library
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import cross_validate
#Define a Reader object
#The Reader object helps in parsing the file or dataframe containing ratings
reader = Reader()

#Create the dataset to be used for building the filter
data = Dataset.load_from_df(ratings, reader)

#Define the algorithm object; in this case kNN
knn = KNNBasic()

#Evaluate the performance in terms of RMSE
cross_validate(knn, data, measures=['RMSE'],verbose=True)
