#!/usr/bin/env python3
from globals import FILES_PATH,representation,embeddings_path
import tensorflow as tf
import numpy as np
from dkm_work.utils import read_list, load_dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp


file_path=FILES_PATH["classic4"]
df_name="classic4"
test_path=f"/home/khaldi/Documents/text-clustering/dkm_work/split/{df_name}/test"
validation_path=f"/home/khaldi/Documents/text-clustering/dkm_work/split/{df_name}/validation"

# Fetch the dataset
dataset,classes = load_dataset(file_path,method=representation,embeddings_path=embeddings_path)
print("Dataset loaded...")
data = dataset.data #matrix n*d
target = dataset.target #matrix n* number_categories

# Get the split between training/test set and validation set
test_indices = read_list(test_path) #array of indices
n_test = test_indices.shape[0]
validation_indices = read_list(validation_path) #array of indices
n_validation = validation_indices.shape[0]

# Filter the dataset
## Keep only the data points in the test and validation sets
test_data = data[test_indices]
test_target = target[test_indices]
validation_data = data[validation_indices]
validation_target = target[validation_indices]
data = sp.vstack([test_data, validation_data])
target = sp.vstack([test_target, validation_target])
## Update test_indices and validation_indices to fit the new data indexing
test_indices = np.asarray(range(0, n_test)) # Test points come first in filtered dataset
validation_indices = np.asarray(range(n_test, n_test + n_validation)) # Validation points come after in filtered dataset


data = data.toarray() # convert data to an array

#load target (ndarray)
M=csr_matrix.toarray(target) #M: n samples times category
mapping = {i: c for i, c in enumerate(classes)}
target = np.array([np.argmax(row) for row in M])
n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = len(classes) # Number of clusters to obtain

# Auto-encoder architecture
input_size = data.shape[1]
hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = n_clusters
dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names
