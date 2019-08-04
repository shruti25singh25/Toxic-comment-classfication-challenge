#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:18:39 2019

@author: shruti
"""

import pandas as pd
import numpy as np
from collections import defaultdict

from constants import LABEL_COLS
from functions import load_bad_words, build_data_path, print_report
from sklearn.model_selection import train_test_split
from random import random

training_data_path = build_data_path('train.csv')
print(training_data_path)

def non_toxic(row):
    return int(all(cell == 0 for cell in row))

df = pd.read_csv(training_data_path)
X = df['comment_text']
y = df[LABEL_COLS]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)

label_counts = y_train.sum()

total = label_counts.sum()

label_freqs = defaultdict(lambda: 0)
for key in label_counts.index:
    label_freqs[key] = label_counts[key]/total
    
def predict(X_values):
    predictions = []
    for example in X_values:
        prediction = []
        for key in LABEL_COLS:
            rand_value = random()
            frequency = label_freqs[key]
            prediction.append(1 if rand_value < frequency else 0)
        predictions.append(prediction)  
    return np.array(predictions)


random_predictions = predict(X_valid)

print('Baseline Data')
print_report(y_valid, random_predictions)
print()
