!pip install -U sentence-transformers



# Import the libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer



# Define some important variables
english_words = set(words.words())
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Essential preprocessing for all the models.
def tokenizer(text):

	"""Returns the processed text."""  

    text = re.sub(r'[^\w\s]', '', text.lower()) # Keep only words and spaces
    tokens = [token for token in text.split() if lemmatizer.lemmatize(token) in english_words \
    		  and token not in stop_words and len(token) > 2] # Keep the english words, delete stopwords, and words of length less than 3
    return ' '.join(tokens)



def document_term_matrix(inputs, vectorizer, min_df, max_df):

	"""Returns the model used for the creation of the document term matrix and the document term matrix.

	    Arguments:
			inputs: A list of lists of documents.
			vectorizer: 'cv' for CountVectorizer, 'tfidf' for TfidfVectorizer.
			min_df: Minimum number of documents that contain a word.
			max_df: Maximum number of documents that contain a word.  

		Returns:
			model: The selected model.
			dtm.toarray(): The sparse matrix transformed into a numpy array.
	"""

    if vectorizer == 'cv':         
        model = CountVectorizer(min_df=min_df, max_df=max_df)  
        dtm = model.fit_transform(inputs)    
    else:        
        model = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=None) 
        dtm = model.fit_transform(inputs)
        
    return model, dtm.toarray()



# Necessary for evaluating the several models.
def get_dictionary(model, articles, min_df, size):

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



# Deep Learning models preprocessing.
def dataset(dtm, batch_size):

	"""Creates the input for ProdLDA, BAT, ETM models.

		Arguments:

			dtm: An array representing the document term matrix.
			batch_size: Number of documents in each batch during model's training.

		Returns:

			train_loader: An iterable over the dataset.			 
	"""

    X_tensor = torch.FloatTensor(dtm)
    train_data = TensorDataset(X_tensor, X_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size)       
    return train_loader    



# Contextualized topic models preprocessing.
def sBert_embeddings(documents, device):

	"""Returns embeddings for the documents in the corpus.

	Arguments:

		documents: A list of lists of documents.
		device: 'cpu' or 'cuda'

	Returns:
	
		sent_embeddings: A list of lists of embeddings given by sBert.

	"""

	model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens').to(device)
	sent_embeddings = model.encode(documents)
	return sent_embeddings



def dataset_creation(dtm, sent_embeddings):

	"""Creates the input for CTM model.

		Arguments:

			dtm: An array representing the document term matrix.
			sent_embeddings: The embeddings given by sBert.

		Returns:

			dataset: A list of dictionaries. The dictionary's keys are the vectors of the document given from CountVectorizer and its values are the respective sBert embeddings.
	"""

    dataset = []

    for i, j in zip(dtm, sent_embeddings):
        dataset.append({'X':i, 'X_bert':torch.FloatTensor(j)})

    return dataset	