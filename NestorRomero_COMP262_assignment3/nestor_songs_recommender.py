# COMP262 - NLP AND RECOMMENDER SYSTEMS
# NESTOR ROMERO - 301133331
# ASSIGNMENT 3 - EXERCISE 2

from optparse import TitledHelpFormatter
import numpy as np
import pandas as pd
import os, gzip, re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
# Dot product calculation > Cosine Similarity
from sklearn.metrics.pairwise import linear_kernel

### FUNCTIONS TO LOAD DATASET
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def load_data():
    # Read original data set    
    data = getDF(os.path.join(os.getcwd(), 'NestorRomero_COMP262_assignment3','meta_Digital_Music.json.gz'))
    return data
### FUNCTIONS TO LOAD DATASET
 
def preprocess_title(row):
    """Function to preprocess the title column

    Args:
        row (_type_): dataset row
    """
    title = row['title']
    title = title.strip().lower()
    title = re.sub(r'[^\w\s]','', title)
    row['title'] = title
    # print(title)

def preprocess_music_data(data):
    """Function to preprocess the dataset and remove unnecesary data

    Args:
        data (dataframe): dataset

    Returns:
        dataframe: preprocessed dataframe
    """
    # Remove unnecesary columns
    columns = data.columns
    columns_to_remove = ['category', 'tech1', 'fit', 'tech2', 'feature', 'rank', 'main_cat', 'similar_item', 'date', 'price', 'imageURL', 'imageURLHighRes']
    keep_filter = [ c not in columns_to_remove for c in columns]
    music_data = data[columns[keep_filter]]
    
    # Text basic preprocessing for remaining data
    music_data.apply(lambda x: preprocess_title(x), axis=1)
    
    # Preprocess title column 
    music_data['title'].replace('', np.nan, inplace=True)
    music_data.dropna(subset=['title'], inplace=True)
    
    # Remove <span <h1 and empty data
    span_rows = music_data.query('title.str.contains("<span")')
    music_data.drop(span_rows.index)
    h1_rows = music_data.query('title.str.contains("<h1")')
    music_data.drop(h1_rows.index)
    
    return music_data    

def create_recommendations( music_data ):
    """Function to create the tfidf matrix for the input dataframe and store the results in a pickle file

    Args:
        music_data (dataframe): Dataframe with records to include in the transformation
    """
    # Calculate tfidf
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(music_data['title'])
    cosine_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    with open(os.path.join(os.getcwd(), 'NestorRomero_COMP262_assignment3','similarity.pckl'), "wb") as f:
        # Pickle the cosine_sim data using the highest protocol available.
        pickle.dump(cosine_matrix, f, pickle.HIGHEST_PROTOCOL)
    

def load_similarities():
    """Function to load similarities (tfidf) matrix from disk

    Returns:
        matrix: tfidf score matrix
    """
    cos_similarities = []
    with open(os.path.join(os.getcwd(), 'NestorRomero_COMP262_assignment3','similarity.pckl'), 'rb') as f:
        cos_similarities = pickle.load(f)
        print(cos_similarities.shape) 
    return cos_similarities
    
def recommend_songs(song_title, data, cos_similarities):
    """Function to create a list of recommendations for a given song title

    Args:
        song_title (str): Name of the song
        data (dataframe): dataset for search recommendations
        cos_similarities (matrix): tfidf scores matrix

    Returns:
        list: recommendations
    """
    song_title = str(song_title).lower().strip()
    
    title_keys = pd.Series(data.index, index=data['title']).drop_duplicates()
    # print(title_keys[21:41])
    
    # Validate if song title is in dataset
    idx = []
    try:
        idx = title_keys[song_title]
        sim_scores = list(enumerate(cos_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)    
    except Exception as ke:
        print(ke)
        return []
    
    # Select top 10 song's scores
    sim_scores = sim_scores[1:11]
    song_indexes = [i[0] for i in sim_scores]

    # Query dataset for similar songs by index
    return data['title'].iloc[song_indexes]
    
    
# MAIN PROGRAM EXECUTION
# LOAD DATASET
data = load_data()
data = preprocess_music_data(data)
# CREATE RECOMMENDATIOKNS > SIMILARITIES MATRIX
### THIS FUNCTION IS ONLY REQUIRED THE FIRST TIME
# create_recommendations(data)

# LOAD SIMILARITIES MATRIX FROM DISK
cos_similarities = load_similarities()

song_title = ''

while True:
    
    # Clear console for better readability
    os.system('cls')
    
    print('Type a song name for recommendations (i.e XYZ)')
    print()
    song_title = input('(Type "Exit" to finish) << : ')
    
    # Exit condition
    if str(song_title).lower().strip() == 'exit':
        break
    
    rec_songs = recommend_songs(song_title, data, cos_similarities)
    if len(rec_songs) > 0:
        print('Here are your recommended songs')    
        for song in rec_songs:
            print(song)
        print()
    else:
        print(f'We have no reccomendations for {song_title}')
    
    input(f'Type any key to continue')
    
print('END OF PROGRAM')
  