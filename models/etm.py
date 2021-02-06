# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class ETM(nn.Module):

    def __init__(self, vocab_size, num_topics, batch_size, word2vec, device):
        super(ETM, self).__init__()
        self.batch_size = batch_size
        self.device = device

        if word2vec == True:
            self.rho = nn.Embedding.from_pretrained(weight)
            self.alphas = nn.Linear(100, num_topics, bias=False)
        else:
            self.rho = nn.Embedding.from_pretrained(weight)
            self.alphas = nn.Linear(300, num_topics, bias=False)

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, 800),
            nn.Softplus(),
            nn.Linear(800, 800),
            nn.Softplus()
        )

        self.mean = nn.Linear(800, num_topics)
        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.logvar = nn.Linear(800, num_topics)
        self.logvar_bn = nn.BatchNorm1d(num_topics)

        self.decoder_bn = nn.BatchNorm1d(vocab_size)

    def encode(self, inputs):
        x = self.encoder(inputs)
        
        posterior_mean = self.mean_bn(self.mean(x))
        posterior_logvar = self.logvar_bn(self.logvar(x))

        KL_divergence = 0.5*torch.sum(1 + posterior_logvar - posterior_mean**2 - torch.exp(posterior_logvar), 1)
        
        return posterior_mean, posterior_logvar, torch.mean(KL_divergence)

    def reparameterization(self, posterior_mean, posterior_logvar):
        epsilon = torch.randn_like(posterior_logvar, device=self.device)
        z = posterior_mean + torch.sqrt(torch.exp(posterior_logvar))*epsilon

        return z

    def get_beta(self):
        beta = F.softmax(self.alphas(self.rho.weight), 0)
        return beta.transpose(1, 0)

    def get_theta(self, normalized_inputs):
        mean, logvar, KL_divergence = self.encode(normalized_inputs)
        z = self.reparameterization(mean, logvar)
        theta = F.softmax(z, 1)

        return theta, KL_divergence

    def decode(self, theta, beta):
        result = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), 1)
        prediction = torch.log(result)

        return prediction

    def forward(self, inputs, normalized_inputs):
        batch_size = inputs.shape[0] 
        if self.batch_size != batch_size:
            self.batch_size = batch_size

        beta = self.get_beta()
        theta, KL_divergence = self.get_theta(normalized_inputs)
        output = self.decode(theta, beta)

        reconstruction_loss = - torch.sum(output*inputs, 1)

        return output, torch.mean(reconstruction_loss) + KL_divergence


def train_model(train_loader, model, optimizer, epochs, device):

	"""Return a list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.

		Arguments:
			
			train_loader: An iterable over the dataset.
			model: The ETM model.
			optimizer: The optimizer for updating ETM's paratemeters.
			epochs: The number of the training iterations.
			device: 'cuda' or 'cpu'.

		Returns:
			
			train_losses: A list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.
			
	"""

    model.train()
    
    for epoch in range(epochs):
        losses = []
        train_losses = []
        total = 0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            normalized_inputs = (inputs/(torch.sum(inputs, 1).unsqueeze(1))).to(device)

            model.zero_grad()
            output, loss = model(inputs, normalized_inputs)
            loss.backward()
            optimizer.step()

            losses.append(loss)
            total += 1

        epoch_loss = sum(losses)/total
        train_losses.append(epoch_loss)        
        print(f'Epoch {epoch + 1}/{epochs}, Loss={epoch_loss}')

        return train_losses


def get_topics(model, tfidf, num_topics):

	"""Returns a list of lists of the top 10 words for each topic.

		Arguments:
			
			model: The ETM model.
			tfidf: The TfidfVectorizer from preprocessing.py.
			num_topics: The number of topics.

		Returns:

			topic_list: A list of lists containing the top 10 words for each topic.
	
	"""

# Generate the topic-word matrix
    beta = model.get_beta()
# Create a list of lists of the top 10 words for each topic  
    topic_list = []

    for topic in beta:
        topic_list.append([tfidf.get_feature_names()[j] for j in topic.argsort()[-10:]])

# Save the resulted list of lists of words for each topic setting
    df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
    df.to_excel(f'ETM_{num_topics}.xlsx')

    return topic_list