# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF as model


def NMF(dtm, tfidf, num_topics):

	"""Returns a list of lists of the top 10 words for each topic.

		Arguments:

			dtm: An array representing the document term matrix.
			tfidf: TfidfVectorizer from preprocessing.py.
			num_topics: The number of topics used by LDA.

		Returns:

			topic_list: A list of lists containing the top 10 words for each topic.
	"""

	nmf = model(n_components=num_topics, max_iter=500, random_state=101)
# Fit the model	
	nmf.fit(dtm)

# Generate the topic-word matrix
	topics = lda.components_
# Create a list of list of the top 10 words for each topic
	topic_list = []

	for topic in topics:
		topic_list.append([tfidf.get_feature_names()[j] for j in topic.argsort()[-10:]])

# Save the resulted list of lists of words for each topic setting
	df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
	df.to_excel(f'C:\\Users\\gxara\\HeinOnline\\NMF_{num_topics}.xlsx')

	return topic_list
