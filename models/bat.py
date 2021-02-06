import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset



class Encoder(nn.Module):


    def __init__(self, vocab_size, num_topics, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size


        self.model = nn.Sequential(
            nn.Linear(vocab_size, 100),
            nn.BatchNorm1d(100, affine=False), # No trainable parameters
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, num_topics),
            nn.Softmax(1)
        )


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        if self.batch_size != batch_size:
            self.batch_size = batch_size 


        x = self.model(inputs)
        return x



class Generator(nn.Module):


    def __init__(self, vocab_size, num_topics, batch_size):
        super(Generator, self).__init__()
        self.batch_size = batch_size


        self.model = nn.Sequential(
            nn.Linear(num_topics, 100),
            nn.BatchNorm1d(100, affine=False), # No trainable parameters
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, vocab_size),
            nn.Softmax(1)
        )


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        if self.batch_size != batch_size:
            self.batch_size = batch_size 


        x = self.model(inputs)
        return x



class Discriminator(nn.Module):


    def __init__(self, vocab_size, num_topics, batch_size):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size


        self.model = nn.Sequential(
            nn.Linear(vocab_size + num_topics, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        if self.batch_size != batch_size:
            self.batch_size = batch_size


        x = self.model(inputs)
        return x



def train_model(epochs, num_topics, n_critic, device):

	"""Return a list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.

		Arguments:

			epochs: The number of the training iterations.
			num_topics: The number of topics.
			n_critic: The number of discriminator iterations per generator iteration
			device: 'cpu' or 'cuda'.

		Returns:
			
			train_losses: A list of lists each containing the Discriminator's, Generator's and Encoder's loss, respectively.
			
	"""

    for epoch in range(epochs):
        losses_d, losses_g, losses_e = [], [], []
        total_losses = []
        total, total_i = 0, 0

        
        for i, (d_r, _) in enumerate(train_loader):
            d_r = (d_r/(torch.sum(d_r, 1).unsqueeze(1))).to(device) # Normalize the inputs
            dirichlet = torch.distributions.Dirichlet(torch.ones(size=(d_r.shape[0], num_topics)))
            theta_f = (dirichlet.sample()).to(device)


            d_f, theta_r = generator(theta_f).detach(), encoder(d_r).detach()
            p_r, p_f = (torch.cat((theta_r, d_r), 1)).to(device), (torch.cat((theta_f, d_f), 1)).to(device)


            discriminator.zero_grad()
            L_d = torch.mean(discriminator(p_f)) - torch.mean(discriminator(p_r))
            L_d.backward()
            optimizer_d.step()


            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01) # Clip the weights of the discriminator


            if i%n_critic == 0:
                generator.zero_grad()
                L_g = - torch.mean(discriminator(p_f))
                L_g.backward()
                optimizer_g.step()
                losses_g.append(L_g)


                encoder.zero_grad()
                L_e = torch.mean(discriminator(p_r))
                L_e.backward()
                optimizer_e.step()
                losses_e.append(L_e)


                losses_d.append(L_d)
                total += 1


        epoch_d = sum(losses_d)/total
        epoch_g = sum(losses_g)/total
        epoch_e = sum(losses_e)/total
        train_losses.append([epoch_d, epoch_g, epoch_e])
        print(f'Epoch {epoch + 1}/{epochs}, Encoder Loss:{epoch_e}, Generator Loss:{epoch_g}, Discriminator Loss:{epoch_d}')


    return train_losses



def get_topics(tfidf, model, num_topics):

	"""Returns a list of lists of the top 10 words for each topic.

		Arguments:

			tfidf: TfidfVectorizer from preprocessing.py.
			topics: The topic-word matrix given by the model.
			num_topics: The number of topics.

		Returns:

			topic_list: A list of lists containing the top 10 words for each topic.
	
	"""

    onehot_topic = torch.eye(num_topics, device=device)
    topic_word_matrix = model(onehot_topic)
    topic_list = []


    for topic in topic_word_matrix:        
        topic_list.append([tfidf.get_feature_names()[j] for j in topic.argsort()[-10:]])


    df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
    df.to_excel(f'BAT_{num_topics}.xlsx')


    return topic_list