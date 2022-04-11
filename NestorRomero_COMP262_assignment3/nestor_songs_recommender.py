# COMP262 - NLP AND RECOMMENDER SYSTEMS
# NESTOR ROMERO - 301133331
# ASSIGNMENT 3 - EXERCISE 2

import numpy as np
import pandas as pd
import os

import pandas as pd
import gzip

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

df = getDF('meta_Digital_Music.json.gz')
print(df.head())