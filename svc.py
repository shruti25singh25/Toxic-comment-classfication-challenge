#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:20:00 2019

@author: shruti
"""

from os import path

from matplotlib import pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

from functions import load_bad_words, build_data_path, print_report
from constants import LABEL_COLS

training_data_path = build_data_path('train.csv')

df = pd.read_csv(training_data_path)

X = df['comment_text']
y = df[LABEL_COLS]
#print(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)


clf = OneVsRestClassifier(SVC(gamma=0.1))

tfidf = TfidfVectorizer(lowercase=True, stop_words='english')
#print(X)
#print(y)

pipeline = make_pipeline(tfidf, clf)

#
pipeline.fit(X_train, y_train)


y_predictions = pipeline.predict(X_valid)
print_report(y_valid, y_predictions)
print('done')
