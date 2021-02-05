import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import pickle
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
from Evaluation.EvaluationMetrics import CoherenceScores


def training(dtm, num_topics):

	"""Returns a list of lists of the top 10 words for each topic.

		Arguments:

			dtm: An array representing the document term matrix.
			num_topics: The number of topics used by LDA.

		Returns:

			topic_list: A list of lists containing the top 10 words for each topic.
	"""

	lda = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=50/num_topics, topic_word_prior=0.01, random_state=101)
	lda.fit(dtm)


# Generate the topic-word matrix
	topics = lda.components_
# Create a list of list of the top 10 words for each topic
	topic_list = []


	for topic in topics:
		topic_list.append([cv.get_feature_names()[j] for j in topic.argsort()[-10:]])


# Save the resulted list of lists of words for each topic setting
	df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
	df.to_excel(f'C:\\Users\\gxara\\Downloads\\LDA_{num_topics}.xlsx')


	return topic_list


if __name__ == '__main__':
# Load the files taken by preprocessing.py
	dtm = np.load('C:\\Users\\gxara\\Downloads\\HeinOnline\\Official Files\\Small\\cv_dtm.npy', allow_pickle=True)
	cv = pickle.load(open('C:\\Users\\gxara\\Downloads\\HeinOnline\\Official Files\\Small\\cv.pickle', 'rb'))
	w2v = KeyedVectors.load('C:\\Users\\gxara\\Downloads\\HeinOnline\\Official Files\\Small\\vectors.kv')
	dictionary = Dictionary.load_from_text('C:\\Users\\gxara\\Downloads\\HeinOnline\\Official Files\\Small\\dictionary.txt')
	bow = np.load('C:\\Users\\gxara\\Downloads\\HeinOnline\\Official Files\\Small\\bow.npy', allow_pickle=True)


	coherence_scores = []

# Run LDA for the topic settings [20, 30, 50, 75, 100]
	for num_topics in [20, 30, 50, 75, 100]:
		print("Starting", "."*num_topics)
		topic_list = training(dtm, num_topics)


# Calculate the C_V, NPMI, UCI and C_W2V measure for each topic setting
		coherence = CoherenceScores(topic_list, bow, dictionary, w2v)
		results = coherence.get_coherence_scores()
		coherence_scores.append(results)
		print(coherence_scores)


# Save the average coherence scores of each topic setting per evaluation metric, and the coherence scores of all the topic settings  
	np.save('C:\\Users\\gxara\\Downloads\\LDA_avg_scores', np.mean(coherence_scores, 0))
	np.save('C:\\Users\\gxara\\Downloads\\LDA_scores', coherence_scores)