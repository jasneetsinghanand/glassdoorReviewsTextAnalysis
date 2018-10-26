#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:55:35 2017

@author: jasneet
"""


# Twitter Classifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

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
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.most_informative_features(15)

# Save Naive Bayes example -->
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#classifier_f = open("naivebayes.pickle", "rb")
#pickle.load(classifier_f)
#classifier_f.close()

#==============================================================================
# 
#==============================================================================
multinomial_classifier = SklearnClassifier(MultinomialNB())
multinomial_classifier.train(training_set)
print("MNB classifier accuracy percent: ", (nltk.classify.accuracy(multinomial_classifier, testing_set))*100)

save_classifier = open("multinomial_classifier.pickle","wb")
pickle.dump(multinomial_classifier, save_classifier)
save_classifier.close()

#==============================================================================
# 
#==============================================================================
bernoulli_classifier = SklearnClassifier(BernoulliNB())
bernoulli_classifier.train(training_set)
print("Bernoulli classifier accuracy percent: ", (nltk.classify.accuracy(bernoulli_classifier, testing_set))*100)

save_classifier = open("bernoulli_classifier.pickle","wb")
pickle.dump(bernoulli_classifier, save_classifier)
save_classifier.close()

#==============================================================================
# 
#==============================================================================
logres_classifier = SklearnClassifier(LogisticRegression())
logres_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(logres_classifier, testing_set))*100)

save_classifier = open("logres_classifier.pickle","wb")
pickle.dump(logres_classifier, save_classifier)
save_classifier.close()

#==============================================================================
# 
#==============================================================================
linearsvc_classifier = SklearnClassifier(LinearSVC())
linearsvc_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(linearsvc_classifier, testing_set))*100)

save_classifier = open("linearsvc_classifier.pickle","wb")
pickle.dump(linearsvc_classifier, save_classifier)
save_classifier.close()

#==============================================================================
# 
#==============================================================================
stochastic_classifier = SklearnClassifier(SGDClassifier())
stochastic_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(stochastic_classifier, testing_set)*100)

save_classifier = open("stochastic_classifier.pickle","wb")
pickle.dump(stochastic_classifier, save_classifier)
save_classifier.close()
