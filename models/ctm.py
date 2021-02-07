# Import the necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from contextualized_topic_models.models.ctm import CombinedTM
from preprocessing import tokenizer, document_term_matrix, get_dictionary, sBert_embeddings, dataset_creation
from evaluation.metrics import CoherenceScores


def CTM(dataset, cv, vocab_size, bert_size, num_topics, top_words, epochs):

	"""Returns the topic list and the document-topic matrix.

		Arguments:

			dataset: Dataset for CTM.
			cv: The CountVectorizer from preprocessing.py.
			vocab_size: The size of the vocabulery.
			bert_size: The size of the sBert embeddings.
			num_topics: The number of topics.
			top_words: The number of the top words for each topics.
			epochs: The number of the training iterations.

		Returns:

			topic_list: A list of lists containint the top 10 words for each topic.
			doc_topic_matrix: A matrix containing the proportion of topics per document.

	"""

	ctm = CombinedTM(input_size=vocab_size, bert_input_size=bert_size, n_components=num_topics, num_epochs=epochs)
	ctm.fit(dataset)

# Generate the topic-word matrix
	word_topic_matrix = ctm.get_topic_word_matrix()
# Generate the topic mixture over the documents
	doc_topic_matrix = ctm.get_doc_topic_distribution(dataset)
# Create a list of lists of the top 10 words for each topic
	topic_list = []

	for topic in word_topic_matrix:
		topic_list.append([cv.get_feature_names()[j] for j in topic.argsort([-top_words:])])

	return topic_list, doc_topic_matrix


if __name__ == '__main__':

# Define the dataset and the arguments
    df = pd.read_csv('HeinOnline.csv')
    articles = df['content']
    min_df = 2
    max_df = 0.7
    num_topics = 20
    size = 100

# Generate the document term matrix and the vectorizer
    processed_articles = articles.apply(tokenizer)
    cv, dtm = document_term_matrix(processed_articles, 'cv', min_df, max_df)
# Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
    bow, dictionary, w2v = get_dictionary(cv, articles, min_df, size)

# Some other arguments
    vocab_size = dtm.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_topics = 20
    bert_size = 768
    epochs = 100
    top_words = 10

# Generate the sBert embeddings
	sent_embeddings = sBert_embeddings(articles, device)

# Create the dataset
	dataset = dataset_creation(dtm, sent_embeddings)

# Generate the list of lists of the top 10 words of each topic and the proportion of topics over the documents
    topic_list, doc_topic_matrix = CTM(dataset, cv, vocab_size, bert_size, num_topics, top_words, epochs)

# Calculate the coherence scores
    evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
    coherence_scores = evaluation_model.get_coherence_scores()
# Print the coherence scores C_V, NPMI, UCI, and C_W2V, respectively
    print(coherence_scores)