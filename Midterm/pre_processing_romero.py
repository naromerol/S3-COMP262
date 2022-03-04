# COMP262 - MIDTERM
# NESTOR ROMERO - 301133331

# 1. IMPORT LIBRARIES
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["The girl kicks a cat", "Cat kicks the girl!", "girl eats food.", "<the cat eats fish>"]

for i in range(len(documents)):
    doc = documents[i]
    # 2. CONVERT LOWERCASE
    doc = doc.lower()
    
    # 3. REMOVE STOP WORDS
    doc_token = word_tokenize(doc)
    doc_no_stop = [i for i in doc_token if not i in stop_words]
    documents[i] = ' '.join(doc_no_stop)

# print(documents)
# 4. TFIDF VECTORIZER
Romero_tfidf = TfidfVectorizer()
# 5. FIT TO DATA
rep_tfidf = Romero_tfidf.fit_transform(documents)
# 6. PRINT IDF
print("IDF for all words in the vocabulary:" , Romero_tfidf.idf_)
# 7. PRINT WORDS
print("All words in the vocabulary: ", Romero_tfidf.get_feature_names())
# 8. PRINT TFIDF
print("TFIDF representation for all documents in our corpus\n",rep_tfidf.toarray())