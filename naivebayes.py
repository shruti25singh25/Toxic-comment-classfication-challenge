#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 00:21:25 2019

@author: shruti
"""

from os import path

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import  BernoulliNB, MultinomialNB#, ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

#from functions import load_bad_words, load_ethnic_slurs, build_data_path, print_report, run_on_test_data
from functions import load_bad_words, build_data_path, print_report, run_on_test_data
from constants import LABEL_COLS

import nltk
from nltk import word_tokenize
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer

BAD_WORDS = set(load_bad_words())
#ETHNIC_SLURS = set(load_ethnic_slurs())

training_data_path = build_data_path('augmented_train.csv')

df = pd.read_csv(training_data_path)

X = df['comment_text']
y = df[LABEL_COLS]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)


clf = OneVsRestClassifier(MultinomialNB())

tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), norm='l2')
bad_word_counter = CountVectorizer(vocabulary=BAD_WORDS)
#slur_counter = CountVectorizer(vocabulary=ETHNIC_SLURS)

#union = make_union(tfidf, bad_word_counter, slur_counter)
union = make_union(tfidf, bad_word_counter)
pipeline = make_pipeline(union, clf)
optimizer = pipeline

optimizer.fit(X_train, y_train) 
y_predictions = optimizer.predict(X_valid)

# best_estimator_ = optimizer.best_estimator_
print_report(y_valid, y_predictions)