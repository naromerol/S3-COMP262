# COMP262 - MIDTERM
# NESTOR ROMERO - 301133331

# 1. IMPORT REQUIRED LIBRARIES
from gensim.models import Word2Vec

corpus = [['girl', 'tricks','cat'], ['cat', 'tricks', 'girl'], ['girl', 'eats','food'], ['cat', 'beats','fish']]

def sgm (corpus):
    skipgram_model = Word2Vec(corpus, min_count=1,sg=1)
    # YOUR CODE GOES HERE
    # 2. LIST WORDS
    word_list = list(skipgram_model.wv.vocab)
    print('2. LIST OF WORDS IN CORPUS')
    print(word_list)
    
    
    # 3. MOST SIMILAR WORDS
    for word in word_list: 
        print(f'MOST SIMILAR WORDS FOR: {word}')
        print( skipgram_model.wv.most_similar(word))
        
    print('4. SIMILARITY BETWEEN WORDS')
    try:
        print(f'Similarity between cat and girl: {skipgram_model.wv.similarity("cat","girl")}')
    except KeyError as ke:
        print(ke)
    
    try:
        print(f'Similarity between eats and food: {skipgram_model.wv.similarity("eats","food")}')
    except KeyError as ke:
        print(ke)
    
    try:
        print(f'Similarity between Girl and eats: {skipgram_model.wv.similarity("cat","girl")}')
    except KeyError as ke:
        print(ke)
    
    
def cbm(corpus):
    # CBOW model
    cbow_model = Word2Vec(corpus, min_count=1,sg=0)
    word_list = list(cbow_model.wv.vocab)
    print('2 CBOW. LIST OF WORDS IN CORPUS')
    print(word_list)
    
    
    # 3B. MOST SIMILAR WORDS
    for word in word_list: 
        print(f'MOST SIMILAR WORDS FOR: {word}')
        print( cbow_model.wv.most_similar(word))
        
    print('4 CBOW. SIMILARITY BETWEEN WORDS')
    try:
        print(f'Similarity between cat and girl: {cbow_model.wv.similarity("cat","girl")}')
    except KeyError as ke:
        print(ke)
    
    try:
        print(f'Similarity between eats and food: {cbow_model.wv.similarity("eats","food")}')
    except KeyError as ke:
        print(ke)
    
    try:
        print(f'Similarity between Girl and eats: {cbow_model.wv.similarity("cat","girl")}')
    except KeyError as ke:
        print(ke)
        
# MAIN EXECUTION BLOCK
sgm(corpus)
cbm(corpus)