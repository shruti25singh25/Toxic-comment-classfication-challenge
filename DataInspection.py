#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:41:20 2019

@author: shruti
"""

from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

from functions import build_data_path, load_bad_words
from constants import LABEL_COLS

STOP_WORDS = set(stopwords.words('english'))


training_data_path = build_data_path('train.csv')
print(training_data_path)

def non_toxic(row):
    return int(all(cell == 0 for cell in row))

df = pd.read_csv(training_data_path)
if 'non_toxic' in LABEL_COLS:
    LABEL_COLS.remove('non_toxic')
df['non_toxic'] = df[LABEL_COLS].apply(lambda x: non_toxic(x), axis=1)
X = df['comment_text']
if 'non_toxic' not in LABEL_COLS:
    LABEL_COLS += ['non_toxic']
y = df[LABEL_COLS]
print(y.sum())
y.sum().plot(kind='bar')
plt.ylabel('Number of Examples')
