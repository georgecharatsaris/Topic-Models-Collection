# Topic-Models-Collection

This repository contains implementations of several topic models on the HeinOnline dataset. By changing
the dataset's path and setting the training data to the respective column one can execute the models 
using their datasets. For instance, replace `parser.add_argument('--dataset', type=str, default='HeiOnline.csv', help='the path to the dataset')` and  `articles = df['content']` with  `parser.add_argument('--dataset', type=str, default='you own path', help='the path to the dataset')` and  `articles = df['the respective column']`.

Also, one can choose how many words to keep during the document term matrix's creation by modifying `parser.add_argument('--min_df', type=int, default=2, help='the minimum number of documents containing a word')`, and  `parser.add_argument('--max_df', type=float, default=0.7, help='the maximum number of topics containing a word')`.

Last but not least, by selecting a different number of topics `parser.add_argument('--num_topics', type=int, default=20, help='the number of topics')`  or number of top words `parser.add_argument('--top_words', type=int, default=10, help='the number of top words for each topic')`,  one can train the different models to yield more topics and print more or less than 10 words in each of the topics.

All the models' parameters are the same as they presented in the official papers, except for LDA in which I use `doc_topic_prior=50/num_topics` and `topic_word_prior=0.01` as proposed in this [paper](https://www.pnas.org/content/101/suppl_1/5228.short). 

The vectorizers `parser.add_argument('--vectorizer', type=str, default='cv', help='the CountVectorizer from sklearn')` and `parser.add_argument('--vectorizer', type=str, default='tfidf', help='the TfidfVectorizer from sklearn')`are selected based on the papers, except for the Topic Models in Embedded Spaces, where I use the tfidf vectorizer instead of the bag-of-words vectorizer.

# Dataset

HeinOnline dataset consists of 3857 journal articles obtained from HeinOnline as a result of 
a Data and Text Mining Agreement. The dataset covers literature between 1960 and 2019, and 
these articles are the result of a Boolean search using the keyword ''artificial 
intelligence''. All the included articles contain both keywords at least once.

[[Link]](https://home.heinonline.org/)

# Useful Files

- The Glove embeddings used in this repository can be found on this [webpage](https://nlp.stanford.edu/projects/glove/).
The file is called Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download).
- The binary dtm for the Dynamic Topic Model can be downloaded [here](https://github.com/magsilva/dtm/tree/master/bin).

# Installation

```
git clone https://github.com/georgecharatsaris/Topic-Models-Collection
cd Topic-Models-Collection
pip install -r requirements.txt
```

# Implementations

## Latent Dirichlet Allocation (LDA):

### Authors

David M. Blei, Andrew Y. Ng, Michael I. Jordan

### Abstract

We describe latent Dirichlet allocation (LDA), a generative probabilistic model for 
collections of discrete data such as text corpora. LDA is a three-level hierarchical 
Bayesian model, in which each item of a collection is modeled as a finite mixture over an 
underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an 
underlying set of topic probabilities. In the context of text modeling, the topic 
probabilities provide an explicit representation of a document. We present efficient 
approximate inference techniques based on variational methods and an EM algorithm for 
empirical Bayes parameter estimation. We report results in document modeling, text 
classification, and collaborative filtering, comparing to a mixture of unigrams model and 
the probabilistic LSI model.

[[Paper]](https://jmlr.org/papers/volume3/blei03a/blei03a.pdf) [[Code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/lda.py)

### Run Example

```
cd models
python lda.py
```

## Non-Negative Factorization (NMF):

### Authors

D. Kuang, J. Choo, H. Park

### Abstract

Nonnegative matrix factorization (NMF) approximates a nonnegative matrix by the product of 
two low-rank nonnegative matrices. Since it gives semantically meaningful result that is 
easily interpretable in clustering applications, NMF has been widely used as a clustering 
method especially for document data, and as a topic modeling method.We describe several 
fundamental facts of NMF and introduce its optimization framework called block coordinate 
descent. In the context of clustering, our framework provides a flexible way to extend NMF 
such as the sparse NMF and the weakly-supervised NMF. The former provides succinct 
representations for better interpretations while the latter flexibly incorporate extra 
information and user feedback in NMF, which effectively works as the basis for the visual 
analytic topic modeling system that we present.Using real-world text data sets, we present 
quantitative experimental results showing the superiority of our framework from the 
following aspects: fast convergence, high clustering accuracy, sparse representation, 
consistent output, and user interactivity. In addition, we present a visual analytic system 
called UTOPIAN (User-driven Topic modeling based on Interactive NMF) and show several usage 
scenarios.Overall, our book chapter cover the broad spectrum of NMF in the context of 
clustering and topic modeling, from fundamental algorithmic behaviors to practical visual 
analytics systems.

[[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-09259-1_7) [[Code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/nmf.py)

### Run Example

```
cd models
python nmf.py
```

## Autoencoding Variational Inference (ProdLDA):

### Authors

Akash Srivastava, Charles Sutton

### Abstract

Topic models are one of the most popular methods for learning representations of text, but a 
major challenge is that any change to the topic model requires mathematically deriving a new 
inference algorithm. A promising approach to address this problem is autoencoding variational 
Bayes (AEVB), but it has proven difficult to apply to topic models in practice. We present 
what is to our knowledge the first effective AEVB based inference method for latent Dirichlet 
allocation (LDA), which we call Autoencoded Variational Inference For Topic Model (AVITM). 
This model tackles the problems caused for AEVB by the Dirichlet prior and by component 
collapsing. We find that AVITM matches traditional methods in accuracy with much better 
inference time. Indeed, because of the inference network, we find that it is unnecessary to 
pay the computational cost of running variational optimization on test data. Because AVITM 
is black box, it is readily applied to new topic models. As a dramatic illustration of this, 
we present a new topic model called ProdLDA, that replaces the mixture model in LDA with a 
product of experts. By changing only one line of code from LDA, we find that ProdLDA yields 
much more interpretable topics, even if LDA is trained via collapsed Gibbs sampling.

[[Paper]](https://arxiv.org/abs/1703.01488) [[Code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/prodlda.py)

### Run Example

```
cd models
python prodlda.py
```

## Neural Topic Modeling with Bidirectional Adversarial Training (BAT):

### Authors

Rui Wangy, Xuemeng Huy, Deyu Zhouy, Yulan Hex, Yuxuan Xiongy, Chenchen Yey, Haiyang Xuz

### Abstract

Recent years have witnessed a surge of interests of using neural topic models for automatic
topic extraction from text, since they avoid the complicated mathematical derivations for
model inference as in traditional topic models such as Latent Dirichlet Allocation (LDA).
However, these models either typically assume improper prior (e.g. Gaussian or Logistic 
Normal) over latent topic space or could not infer topic distribution for a given document. 
To address these limitations, we propose a neural topic modeling approach, called 
Bidirectional Adversarial Topic (BAT) model, which represents the first attempt of applying 
bidirectional adversarial training for neural topic modeling. The proposed BAT builds a 
twoway projection between the document-topic distribution and the document-word distribution.
It uses a generator to capture the semantic patterns from texts and an encoder for topic 
inference. Furthermore, to incorporate word relatedness information, the Bidirectional 
Adversarial Topic model with Gaussian (Gaussian-BAT) is extended from BAT. To verify the 
effectiveness of BAT and Gaussian- BAT, three benchmark corpora are used in our experiments. 
The experimental results show that BAT and Gaussian-BAT obtain more coherent topics, 
outperforming several competitive baselines. Moreover, when performing text clustering based 
on the extracted topics, our models outperform all the baselines, with more significant 
improvements achieved by Gaussian-BAT where an increase of near 6% is observed in accuracy.

[[Paper]](https://arxiv.org/abs/2004.12331) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/bat.py)

### Run Example

```
cd models
python bat.py
```

## Topic Models in Embedded Spaces (ETM):

### Authors

Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei

### Abstract

Topic modeling analyzes documents to learn meaningful patterns of words. However, existing 
topic models fail to learn interpretable topics when working with large and heavy-tailed 
vocabularies. To this end, we develop the embedded topic model (ETM), a generative model of 
documents that marries traditional topic models with word embeddings. In particular, it 
models each word with a categorical distribution whose natural parameter is the inner 
product between a word embedding and an embedding of its assigned topic. To fit the ETM, we 
develop an efficient amortized variational inference algorithm. The ETM discovers 
interpretable topics even with large vocabularies that include rare words and stop words. 
It outperforms existing document models, such as latent Dirichlet allocation, in terms of 
both topic quality and predictive performance.

[[Paper]](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00325) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/etm.py)

### Run Example

```
cd models
python etm.py
```

## Contextualized Topic Models (CTM):

### Authors

Federico Bianchi, Silvia Terragni, Dirk Hovy

### Abstract

Topic models extract meaningful groups of words from documents, allowing for a better 
understanding of data. However, the solutions are often not coherent enough, and thus harder 
to interpret. Coherence can be improved by adding more contextual knowledge to the model. 
Recently, neural topic models have become available, while BERT-based representations have 
further pushed the state of the art of neural models in general. We combine pre-trained 
representations and neural topic models. Pre-trained BERT sentence embeddings indeed support 
the generation of more meaningful and coherent topics than either standard LDA or existing 
neural topic models. Results on four datasets show that our approach effectively increases 
topic coherence.

[[Paper]](https://arxiv.org/abs/2004.03974) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/ctm.py) [[GitHub]](https://github.com/MilaNLProc/contextualized-topic-models)

### Run Example

```
cd models
python ctm.py
```

## Dynamic Topic Models (DTM):

### Authors

David M. Blei, John D. Lafferty

### Abstract

A family of probabilistic time series models is developed to analyze the time evolution of 
topics in large document collections. The approach is to use state space models on the 
natural parameters of the multinomial distributions that represent the topics. Variational 
approximations based on Kalman filters and nonparametric wavelet regression are developed 
to carry out approximate posterior inference over the latent topics. In addition to giving 
quantitative, predictive models of a sequential corpus, dynamic topic models provide a 
qualitative window into the contents of a large document collection. The models are 
demonstrated by analyzing the OCR’ed archives of the journal Science from 1880 through 
2000.

[[Paper]](https://dl.acm.org/doi/abs/10.1145/1143844.1143859) [[code]](https://github.com/georgecharatsaris/Topic-Models-Collection/blob/main/models/dtm.py)

### Run Example

```
cd models
python dtm.py
```