## Topic-Models-Collection

This repository contains implementations of several topic models. 

## Implementations

- Latent Dirichlet Allocation (LDA):

## Authors

David M. Blei, Andrew Y. Ng, Michael I. Jordan

## Abstract

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