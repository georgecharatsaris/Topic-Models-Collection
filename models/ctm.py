!pip install contextualized_topic_models
# Import the necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from contextualized_topic_models.models.ctm import CombinedTM


def CTM(dataset, cv, vocab_size, bert_size, num_topics):

	"""Returns the topic list and the document-topic matrix.

		Arguments:

			dataset: Dataset for CTM.
			cv: CountVectorizer from preprocessing.py.
			vocab_size: The size of the vocabulery.
			bert_size: The size of the sBert embeddings.
			num_topics: The number of topics.

		Returns:

			topic_list: A list of lists containint the top 10 words for each topic.
			doc_topic_matrix: A matrix containing the proportion of topics per document.

	"""

	ctm = CombinedTM(input_size=vocab_size, bert_input_size=bert_size, n_components=num_topics)
	ctm.fit(dataset)

# Generate the topic-word matrix
	word_topic_matrix = ctm.get_topic_word_matrix()
# Generate the topic mixture over the documents
	doc_topic_matrix = ctm.get_doc_topic_distribution(dataset)
# Create a list of lists of the top 10 words for each topic
	topic_list = []

	for topic in word_topic_matrix:
		topic_list.append([cv.get_feature_names()[j] for j in topic.argsort([-10:])])

	return topic_list, doc_topic_matrix