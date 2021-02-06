# Topic-Models-Collection

This repository contains implementations of several topic models. 

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

[code](https://github.com/Georgios1993/Topic-Models-Collection/blob/main/models/lda.py)

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

[code](https://github.com/Georgios1993/Topic-Models-Collection/blob/main/models/nmf.py)

## Autoencoding Variational Inference (ProdLDA):

### Authors

Akash Srivastava, Charles Sutton

### Abstract

Topic models are one of the most popular methods for learning representations of
text, but a major challenge is that any change to the topic model requires mathematically
deriving a new inference algorithm. A promising approach to address
this problem is autoencoding variational Bayes (AEVB), but it has proven difficult
to apply to topic models in practice. We present what is to our knowledge the
first effective AEVB based inference method for latent Dirichlet allocation (LDA),
which we call Autoencoded Variational Inference For Topic Model (AVITM). This
model tackles the problems caused for AEVB by the Dirichlet prior and by component
collapsing. We find that AVITM matches traditional methods in accuracy
with much better inference time. Indeed, because of the inference network, we
find that it is unnecessary to pay the computational cost of running variational
optimization on test data. Because AVITM is black box, it is readily applied
to new topic models. As a dramatic illustration of this, we present a new topic
model called ProdLDA, that replaces the mixture model in LDA with a product
of experts. By changing only one line of code from LDA, we find that ProdLDA
yields much more interpretable topics, even if LDA is trained via collapsed Gibbs
sampling.

[code](https://github.com/Georgios1993/Topic-Models-Collection/blob/main/models/prodlda.py)

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

[code](https://github.com/Georgios1993/Topic-Models-Collection/blob/main/models/bat.py)

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

[code](https://github.com/Georgios1993/Topic-Models-Collection/blob/main/models/etm.py)
