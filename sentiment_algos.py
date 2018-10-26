#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 02:19:47 2017

@author: jasneet
"""



# Twitter Classifier
import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode
import statistics
from sentiment_analysis import VoteClassifier

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        try:
            answer = mode(votes)
        except statistics.StatisticsError as e:
            answer = 'neutral'
        return answer

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

documents = []
all_words = []

# filter the data
allowed_word_types = ["J"]

short_pos = open("positive.txt","r", encoding="utf-8", errors="ignore").read()
short_neg = open("negative.txt","r", encoding="utf-8", errors="ignore").read()

for r in short_pos.split('\n'):
    documents.append( (r,'pos') )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append( (r,'neg') )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
        

save_documents = open("docs.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

#==============================================================================
# Find frequency distribution of words and then Create word features    
#==============================================================================
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]
    

save_word_features = open("text_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)



training_set = featuresets[:10000] 
testing_set = featuresets[10000:]

#==============================================================================
# Vannila Naive Bayes
#==============================================================================
# open_file = open("naivebayes.pickle", "rb")
# classifier = pickle.load(open_file)
# open_file.close()

#==============================================================================
# 
#==============================================================================
open_file = open("multinomial_classifier.pickle", "rb")
multinomial_classifier = pickle.load(open_file)
open_file.close()
#==============================================================================
# 
#==============================================================================
open_file = open("bernoulli_classifier.pickle", "rb")
bernoulli_classifier = pickle.load(open_file)
open_file.close()

#==============================================================================
# 
#==============================================================================
open_file = open("logres_classifier.pickle", "rb")
logres_classifier = pickle.load(open_file)
open_file.close()

#==============================================================================
# 
#==============================================================================
open_file = open("linearsvc_classifier.pickle", "rb")
linearsvc_classifier = pickle.load(open_file)
open_file.close()
#==============================================================================
# 
#==============================================================================
open_file = open("stochastic_classifier.pickle", "rb")
stochastic_classifier = pickle.load(open_file)
open_file.close()

most_voted_classifier = VoteClassifier(
                                  linearsvc_classifier,
                                  multinomial_classifier,
                                  bernoulli_classifier,
                                  logres_classifier,
                                  stochastic_classifier)

def sentiment(text):
    feats = find_features(text)
    return most_voted_classifier.classify(feats),most_voted_classifier.confidence(feats)

