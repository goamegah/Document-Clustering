PARAMS={
    # Parameter setting from arguments
    "p_epochs" : 50, #Number of pretraining epochs
    "f_epochs" : 50, #Number of fine-tuning epochs per alpha value
    "lambda": 1.0, #Value of the hyperparameter weighing the clustering loss against the reconstruction loss
    "batch_size" : 256, # Size of the mini-batches used in the stochastic optimizer
    "validation" : False, # Specify if data should be split into validation and test sets
    "pretrain" : True, # Specify if DKM's autoencoder should be pretrained
    "annealing" : False, # Specify if annealing should be used
    "seeded" : True,# Specify if runs are seeded,
    "cpu":True

}

FILES_PATH={
    "classic3":"/home/khaldi/Downloads/classic3.csv",
    "classic4":"/home/khaldi/Downloads/classic4.csv",
    "bbc":"/home/khaldi/Downloads/bbc.csv"
}

representation="glove"
embeddings_path="/home/khaldi/glove.840B.300d.txt"


