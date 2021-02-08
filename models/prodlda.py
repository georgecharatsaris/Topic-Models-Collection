# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from preprocessing import tokenizer, document_term_matrix, get_dictionary, dataset
from evaluation.metrics import CoherenceScores


class ProdLDA(nn.Module):

	def __init__(self, vocab_size, num_topics, batch_size, device):
		super(ProdLDA, self).__init__()
		self.batch_size = batch_size
		self.num_topics = num_topics
		self.vocab_size = vocab_size
		self.device = device

		self.encoder_input = nn.Linear(vocab_size, 100)
		self.encoder_lr = nn.Linear(100, 100)
		self.encoder_drop = nn.Dropout(0.2)

		self.posterior_mean = nn.Linear(100, num_topics)
		self.posterior_mean_bn = nn.BatchNorm1d(num_topics, affine=False) # No trainable parameters
		self.posterior_logvar = nn.Linear(100, num_topics)
		self.posterior_logvar_bn = nn.BatchNorm1d(num_topics, affine=False) # No trainable parameters

		self.beta = (torch.Tensor(num_topics, vocab_size)).to(device)
		self.beta = nn.Parameter(self.beta)
		nn.init.xavier_uniform_(self.beta)
		self.beta_bn = nn.BatchNorm1d(vocab_size, affine=False) # No trainable parameters
		self.decoder_drop = nn.Dropout(0.2)

	def encoder(self, inputs):
		encoder_input = F.softplus(self.encoder_input(inputs))
		encoder_lr = F.softplus(self.encoder_lr(encoder_input))
		encoder_drop = self.encoder_drop(encoder_lr)

		posterior_mean = self.posterior_mean_bn(self.posterior_mean(encoder_drop))
		posterior_logvar = self.posterior_logvar_bn(self.posterior_logvar(encoder_drop))
		posterior_var = (torch.exp(posterior_logvar)).to(self.device)
		prior_mean = (torch.tensor([0.]*self.num_topics)).to(self.device)
		prior_var = (torch.tensor([1. - 1./self.num_topics]*self.num_topics)).to(self.device)
		prior_logvar = (torch.log(prior_var)).to(self.device)  
		KL_divergence = 0.5*torch.sum(posterior_var/prior_var + ((prior_mean - posterior_mean)**2)/prior_var \
		  						  - self.num_topics + (prior_logvar) - posterior_logvar, 1)
   
		return posterior_mean, posterior_logvar, KL_divergence

	def reparameterization(self, mu, std):
		epsilon = (torch.randn_like(std)).to(self.device)
		sigma = (torch.sqrt(torch.exp(std))).to(self.device)
		z = mu + sigma*epsilon
		return z
   
	def decoder(self, z):
		theta = self.decoder_drop(F.softmax(z, 1))
		word_dist = F.softmax(self.beta_bn(torch.matmul(theta, self.beta)), 1)
		return word_dist

	def forward(self, inputs):
		batch_size = inputs.shape[0]

		if self.batch_size != batch_size:
			self.batch_size = batch_size
   
		posterior_mean, posterior_logvar, KL_divergence = self.encoder(inputs)
		z = self.reparameterization(posterior_mean, posterior_logvar)
		self.topic_word_matrix = self.beta
		word_dist = self.decoder(z)
		reconstruction_loss = - torch.sum(inputs*torch.log(word_dist), 1) 
		  
		return word_dist, torch.mean(KL_divergence + reconstruction_loss)


def train_model(train_loader, model, optimizer, epochs, device):

	"""Trains the model.

		Arguments:

			train_loader: An iterable over the dataset.
			model: The ProdLDA model.
			optimizer: The optimizer for updating ProdLDA's paratemeters.
			epochs: The number of the training iterations.
			device: 'cuda' or 'cpu'.

		Returns:

			train_losses: A list of the model's losses during the training. 

		"""

	model.train()
	train_losses = []

	for epoch in range(epochs):
		losses = []
		total = 0

		for inputs, _ in train_loader:
			inputs = inputs.to(device)
			model.zero_grad()
			output, loss = model(inputs)
			loss.backward()
			optimizer.step()
			losses.append(loss)
			total += 1

		epoch_loss = sum(losses)/total
		train_losses.append(epoch_loss)   
		print(f'Epoch {epoch + 1}/{epochs}: Loss={epoch_loss}')

	return train_losses


def get_topics(cv, model, num_topics, top_words):

	"""Returns a list of lists of the top words for each topic.

		Arguments:

			cv: The CountVectorizer from preprocessing.py.
			model: The ProdLDA model.
			num_topics: The number of topics.
			top_words: The number of the top words for each topics.

		Returns:

			topic_list: A list of lists containing the top words for each topic.
	"""

# Generate the topic-word matrix
	topics = model.topic_word_matrix
# Create a list of lists of the top words for each topic
	topic_list = []
	  
	for topic in topics:
		topic_list.append([cv.get_feature_names()[j] for j in topic.argsort()[-top_words:]])

# Save the resulted list of lists of words for each topic setting
	df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
	df.to_excel(f'ProdLDA_{num_topics}.xlsx')

	return topic_list


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
	epochs = 100
	top_words = 10

# Create the train loader
	train_loader = dataset(dtm, batch_size)

# Define the model and the optimizer
	prodLDA = ProdLDA(vocab_size, num_topics, batch_size, device).to(device)
	optimizer = optim.Adam(prodLDA.parameters(), lr=0.002, betas=(0.9, 0.999))

# Train the model
	train_model(train_loader, prodLDA, optimizer, epochs, device)

# Create the list of lists of the top 10 words of each topic
	topic_list = get_topics(cv, prodLDA, num_topics, top_words)

# Calculate the coherence scores
	evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
	coherence_scores = evaluation_model.get_coherence_scores()
# Print the coherence scores C_V, NPMI, UCI, and C_W2V, respectively
	print(coherence_scores)