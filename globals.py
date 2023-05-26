PARAMS = {
    # Parameter setting from arguments
    "p_epochs": 2500,  # Number of pretraining epochs
    "f_epochs": 300,  # Number of fine-tuning epochs per alpha value
    "lambda": 1.0,
    # Value of the hyperparameter weighing the clustering loss against the reconstruction loss
    "batch_size": 256,  # Size of the mini-batches used in the stochastic optimizer
    "validation": False,  # Specify if data should be split into validation and test sets
    "pretrain": True,  # Specify if DKM's autoencoder should be pretrained
    "annealing": False,  # Specify if annealing should be used
    "seeded": True,  # Specify if runs are seeded,
    "cpu": True

}

FILES_PATH = {
    "classic3": "/home/godwin/Documents/Uparis/M1MLSD2223/ue_data2/text-clustering/core/datasets"
                "/data/bbc.csv",
    "classic4": "/home/godwin/Documents/Uparis/M1MLSD2223/ue_data2/text-clustering/core/datasets"
                "/data/classic4.csv",
    "bbc": "/home/godwin/Documents/Uparis/M1MLSD2223/ue_data2/text-clustering/core/datasets/data"
           "/bbc.csv",
    "bbc_doc2vec": "/home/godwin/Documents/Uparis/M1MLSD2223/ue_data2/text-clustering/core"
                   "/datasets/data/bbc_doc2vec.csv",
    "classic4_doc2vec": "/home/godwin/Documents/Uparis/M1MLSD2223/ue_data2/text-clustering/core"
                        "/datasets/data/classic4_doc2vec.csv",
    "classic3_doc2vec": "/home/godwin/Documents/Uparis/M1MLSD2223/ue_data2/text-clustering/core"
                        "/datasets/data/classic3_doc2vec.csv",
}

representation = "glove"
embeddings_path = "/home/godwin/Documents/Uparis/M1MLSD2223/ue_data2/tmp/glove.840B.300d.txt"
