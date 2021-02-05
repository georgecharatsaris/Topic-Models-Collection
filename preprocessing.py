# Import the libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec, KeyedVectors
import pickle


# Define some important variables
english_words = set(words.words())
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Load the dataset and choose only the articles
df = pd.read_csv('C:\\Users\\gxara\\Downloads\\HeinOnline\\HeinOnline.csv', index_col=0)
articles = df['content']


def tokenizer(text):

	"""Returns the processed text."""  

    text = re.sub(r'[^\w\s]', '', text.lower()) # Keep only words and spaces
    tokens = [token for token in text.split() if lemmatizer.lemmatize(token) in english_words \
    		  and token not in stop_words and len(token) > 2] # Keep the english words, delete stopwords, and words of length less than 3
    return ' '.join(tokens)


def document_term_matrix(inputs, vectorizer, min_df=2):

	"""Returns the model used for the creation of the document term matrix and the document term matrix.

	    Arguments:
			inputs: A list of lists of documents.
			vectorizer: 'cv' for CountVectorizer, 'tfidf' for TfidfVectorizer.
			min_df: Minimum number of documents containing that contain a word.

		Returns:
			model: The selected model.
			dtm.toarray(): The sparse matrix transformed into a numpy array.
	"""

    if vectorizer == 'cv':         
        model = CountVectorizer(min_df=min_df, max_df=0.7) # keep words with frequency less than 70% of the documents and words that appear in at least min_df documents
        dtm = model.fit_transform(inputs)    
    else:        
        model = TfidfVectorizer(min_df=min_df, max_df=0.7) # keep words with frequency less than 70% of the documents and words that appear in at least min_df documents
        dtm = model.fit_transform(inputs)
        
    return model, dtm.toarray()


def further_preprocessing(model, articles, min_df, size):

	"""Returns the bag of words, the dictionary of the corpus, and the w2v vectors of the words in the dictionary.

		Arguments:
			model: CountVectorizer or TfidfVectorizer trained on the articles.
			articles: A list of lists of processed documents (at least without punctuation).
			min_df : The minimum number of documents that contain a word.
			size: The size of w2v vectors.

		Returns:
			bow: A list of lists of tokens.
			dictionary: The dictionary of the corpus.
			w2v: The w2v model built on the dictionary.
	"""

    bow = [[token for token in article.split() if token in model.vocabulary_] for article in articles] 
    dictionary = Dictionary(bow)
    w2v = Word2Vec(bow, size=size, min_count=min_df)
    return bow, dictionary, w2v


processed_articles = articles.apply(tokenizer)
# Define the necessary arguments for the functions
min_df = 2
size = 100


model, dtm = document_term_matrix(processed_articles, 'cv', min_df=min_df)
bow, dictionary, w2v = further_preprocessing(model, processed_articles, min_df, size)


# Save the vectorizer and the document term matrix for later use as the topic models' inputs
np.save('C:\\Users\\gxara\\Downloads\\dtm', dtm) 
pickle.dump(model, open('C:\\Users\\gxara\\Downloads\\cv.pickle', 'wb'))
# Save the bag of words, the dictionary and the w2v vectors for the topic models' evaluation
np.save('C:\\Users\\gxara\\Downloads\\bow', bow)
dictionary.save_as_text('C:\\Users\\gxara\\Downloads\\dictionary.txt')
word_vectors = w2v.wv
word_vectors.save('C:\\Users\\gxara\\Downloads\\vectors.kv')