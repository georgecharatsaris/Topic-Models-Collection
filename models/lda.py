# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


def LDA(dtm, cv, num_topics):

	"""Returns a list of lists of the top 10 words for each topic.

		Arguments:

			dtm: An array representing the document term matrix.
			cv: CountVectorizer from preprocessing.py.
			num_topics: The number of topics used by LDA.

		Returns:

			topic_list: A list of lists containing the top 10 words for each topic.
	"""

	lda = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=50/num_topics, topic_word_prior=0.01, random_state=101)
# Fit the model
	lda.fit(dtm)

# Generate the topic-word matrix
	topics = lda.components_
# Create a list of list of the top 10 words for each topic
	topic_list = []

	for topic in topics:
		topic_list.append([cv.get_feature_names()[j] for j in topic.argsort()[-10:]])

# Save the resulted list of lists of words for each topic setting
	df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
	df.to_excel(f'C:\\Users\\gxara\\HeinOnline\\LDA_{num_topics}.xlsx')

	return topic_list