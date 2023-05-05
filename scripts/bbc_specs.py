#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Godwin AMEGAH"
__license__ = "GPL"

import pandas as pd
from global_vars import datasets_path


# Load the dataset
dataset = pd.read_csv(filepath_or_buffer=f'{datasets_path}/bbc.csv')
print("Dataset BBC loaded...")
documents = dataset.text
labels = dataset.label  # Labels are topic (e.g sport) match document
n_samples = documents.shape[0]  # Number of samples in the dataset
n_clusters = len(list(labels.unique()))  # Number of clusters to obtain
