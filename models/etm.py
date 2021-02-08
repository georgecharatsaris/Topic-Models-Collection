# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pandas as pd
from preprocessing import tokenizer, document_term_matrix, get_dictionary, dataset, glove_embeddings, word2vec_embeddings
from evaluation.metrics import CoherenceScores
from sklearn.preprocessing import normalize


class ETM(nn.Module):

    def __init__(self, vocab_size, num_topics, batch_size, word2vec, weights, device):
        super(ETM, self).__init__()
        self.batch_size = batch_size
        self.device = device

        if word2vec == True:
            self.rho = nn.Embedding.from_pretrained(weights)
            self.alphas = nn.Linear(100, num_topics, bias=False)
        else:
            self.rho = nn.Embedding.from_pretrained(weights)
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


def get_topics(model, tfidf, num_topics, top_words):

    """Returns a list of lists of the top words for each topic.

        Arguments:

            model: The ETM model.
            tfidf: The TfidfVectorizer from preprocessing.py.
            num_topics: The number of topics.
            top_words: The number of the top words for each topics.

        Returns:

            topic_list: A list of lists containing the top words for each topic.

    """

# Generate the topic-word matrix
    beta = model.get_beta()
# Create a list of lists of the top words for each topic  
    topic_list = []

    for topic in beta:
        topic_list.append([tfidf.get_feature_names()[j] for j in topic.argsort()[-top_words:]])

# Save the resulted list of lists of words for each topic setting
    df = pd.DataFrame(np.array(topic_list).T, columns=[f'Topic {i + 1}' for i in range(num_topics)])
    df.to_excel(f'ETM_{num_topics}.xlsx')

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
    tfidf, dtm = document_term_matrix(processed_articles, 'tfidf', min_df, max_df)
    dtm = normalize(dtm)
# Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
    bow, dictionary, w2v = get_dictionary(tfidf, articles, min_df, size)

# Some other arguments
    vocab_size = dtm.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_topics = 20
    epochs = 100
    n_critic = 5
    top_words = 10
    embeddings = 'Word2Vec' # or 'GloVe'

    if embeddings == 'GloVe':
# Load the GloVe embeddings
        embeddings_dict = {}

        with open("glove.42B.300d.txt", 'rb') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

# Create a matrix containing the GloVe embeddings
        embedding_matrix = glove_embeddings(tfidf, vocab_size, embeddings_dict)
    else:
# Create a matrix containing the Word2Vec embeddings
        embedding_matrix = word2vec_embeddings(tfidf, vocab_size, w2v)

# Make the embedding matrix a float tensor to be used as rho's weights
    weights = torch.FloatTensor(embedding_matrix)

# Create the train loader
    train_loader = dataset(dtm, batch_size)

# Define the models and the optimizers
    model = (ETM(vocab_size, num_topics, batch_size, True, weights, device)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), weight_decay=1.2e-6)

# Train the model
    train_model(train_loader, model, optimizer, epochs, device)

# Create the list of lists of the top 10 words of each topic
    topic_list = get_topics(model, tfidf, num_topics, top_words)

# Calculate the coherence scores
    evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
    coherence_scores = evaluation_model.get_coherence_scores()
# Print the coherence scores C_V, NPMI, UCI, and C_W2V, respectively
    print(coherence_scores)