import torch
from torch import nn

#ModelForEmbedding
DEFAULT_PATH_GLOVE_MODEL = "/home/khaldi/modeles_word2vec/model_glove42B"

#InterfaceDeep
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

#ReductionEmbedding
DEFAULT_PARAMS={
    "PCA":
        {
            "n_components":2
        },
    "TSNE":
        {
            "n_components":2,
            "perplexity": 30.
        },
    "AE":
        {
            "id":{
                "device":DEVICE,
                "loss_fn":nn.MSELoss()
            },
            "train":{
                "lr":1e-3,
                "n_epochs":100,
                "batch_size":10,
                "dimension_encoder_out":2
            }
        },
    "UMAP":dict({})

}



