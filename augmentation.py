#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:39:02 2019

@author: shruti
"""

import pandas as pd

raw_df = pd.read_csv('./data/raw/train.csv')
#de_df = pd.read_csv('./data/raw/train.csv')
#es_df = pd.read_csv('./data/raw/train.csv')
#fr_df = pd.read_csv('./data/raw/train.csv')

master_df = raw_df.append(de_df).append(es_df).append(fr_df)

master_df.to_csv('./data/raw/augmented_train.csv')
