# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:08:04 2021

@chapter #6
"""
import numpy as np

import pandas as pd
#Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

users.head()

#Load the u.item file into a dataframe
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

movies.head()

#Remove all information except Movie ID and title
movies = movies[['movie_id', 'title']]

#Load the u.data file into a dataframe
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

ratings.head()

#Drop the timestamp column
ratings = ratings.drop('timestamp', axis=1)
"""
The ratings DataFrame contains user ratings for movies that range from 1 to 5.
 Therefore, we can model this problem as an instance of supervised learning 
 where we need to predict the rating, given a user and a movie.
 Although the ratings can take on only five discrete values, we will model 
 this as a regression problem.

"""
"""
Let's now split our ratings dataset in such a way that 75% of a user's ratings 
is in the training dataset and 25% is in the testing dataset. 
We will do this using a slightly hacky way: we will assume that the user_id field is 
the target variable (or y) and that our ratings DataFrame consists 
of the predictor variables (or X). We will then pass these two variables
 into scikit-learn's train_test_split function and stratify it along y. 
 This ensures that the proportion of each class is the same in both the training 
 and testing datasets
"""
#Import the train_test_split function
from sklearn.model_selection import train_test_split

#Assign X as the original ratings dataframe and y as the user_id column of ratings.
X = ratings.copy()
y = ratings['user_id']

#Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=42)

#Import the mean_squared_error function
from sklearn.metrics import mean_squared_error

#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


#Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model):
    
    #Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])
    #Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in id_pairs])
    #Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])
    #Return the final RMSE score
    return rmse(y_true, y_pred)


#Build the ratings matrix using pivot_table function
"""
each row represents a user and each column represents a movie.
 Therefore, the value in the ith row and jth column will denote
 the rating given by user i to movie j.
"""
r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie_id')

r_matrix.head()
r_matrix.info()
"""
Let's first build one of the simplest collaborative filters possible.
 This simply takes in user_id and movie_id and outputs the mean rating for the movie
 by all the users who have rated it. No distinction is made between the users. In other 
 words, the rating of each user is assigned equal weight.

It is possible that some movies are available only in the test set and not the training 
set (and consequentially, not in our ratings matrix). In such cases, we will just default
 to a rating of 3.0, like the baseline model:
"""

#User Based Collaborative Filter using Mean Ratings
def cf_user_mean(user_id, movie_id):
    
    #Check if movie_id exists in r_matrix
    if movie_id in r_matrix:
        #Compute the mean of all the ratings given to the movie
        mean_rating = r_matrix[movie_id].mean()
    
    else:
        #Default to a rating of 3.0 in the absence of any information
        mean_rating = 3.0
    
    return mean_rating

#Compute RMSE for the Mean model
score(cf_user_mean)



# output 1.0234701463131335
"""
if you need to see what is going on un_comment the below and run and the variable explorer
"""
#id_pairs = zip(X_test['user_id'], X_test['movie_id'])
#list(id_pairs)
#y_true1 = np.array(X_test['rating'])
#y_pred1 = np.array([cf_user_mean(user, movie) for (user, movie) in id_pairs])

#result = rmse(y_true1,y_pred1)

#just to see the output
#true_act =  pd.DataFrame({'y_true':y_true1,'y_predict': y_pred1})



