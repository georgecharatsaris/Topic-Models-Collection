# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from preprocessing import tokenizer, document_term_matrix, get_dictionary
from evaluation.metrics import CoherenceScores
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--vectorizer', type=str, default='cv', help='the CountVectorizer from sklearn')
parser.add_argument('--min_df', type=int, default=2, help='the minimum number of documents containing a word')
parser.add_argument('--max_df', type=float, default=0.7, help='the maximum number of topics containing a word')
parser.add_argument('--size', type=int, default=100, help='the size of the w2v embeddings')
parser.add_argument('--num_topics', type=int, default=20, help='the number of topics')
parser.add_argument('--top_words', type=int, default=10, help='the number of top words for each topic')
opt = parser.parse_args()


def LDA(dtm, cv, num_topics, top_words):

	"""Returns a list of lists of the top words for each topic.

		Arguments:

			dtm: An array representing the document term matrix.
			cv: The CountVectorizer from preprocessing.py.
			num_topics: The number of topics used by LDA.
			top_words: The number of the top words for each topics.

		Returns:

			topic_list: A list of lists containing the top words for each topic.
	"""

	lda = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=50/num_topics, topic_word_prior=0.01, random_state=101)
# Fit the model
	lda.fit(dtm)

# Generate the topic-word matrix
	topics = lda.components_
# Create a list of lists of the top words for each topic
	topic_list = []

	for topic in topics:
		topic_list.append([cv.get_feature_names()[j] for j in topic.argsort()[-top_words:]])

# Save the resulted list of lists of words for each topic setting
	df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
	df.to_excel(f'LDA_{num_topics}.xlsx')

	return topic_list


if __name__ == '__main__':
# Define the dataset and the arguments
	df = pd.read_csv('HeinOnline.csv')
	articles = df['content']

# Generate the document term matrix and the vectorizer
	processed_articles = articles.apply(tokenizer)
	cv, dtm = document_term_matrix(processed_articles, opt.vectorizer, opt.min_df, opt.max_df)
# Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
	bow, dictionary, w2v = get_dictionary(cv, articles, opt.min_df, opt.size)

# Create the list of lists of the top words of each topic
	topic_list = LDA(dtm, cv, opt.num_topics, opt.top_words)

# Calculate the coherence scores
	evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
	coherence_scores = evaluation_model.get_coherence_scores()
# Print the coherence scores C_V, NPMI, UCI, and C_W2V, respectively
	print(coherence_scores)