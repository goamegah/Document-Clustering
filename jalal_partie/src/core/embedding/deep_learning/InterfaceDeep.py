import numpy as np
import torch
from torch import nn, optim
from src.core.embedding.deep_learning.AE import AE
from src.core.globals import DEVICE

class InterfaceDeep:
    def __init__(self, device: str = DEVICE, loss_fn: nn.MSELoss=nn.MSELoss()):
        self.device = device
        self.loss_fn = loss_fn
        self.losses=[]

    def fit(self,X: np.ndarray, lr=1e-3, n_epochs=100, batch_size=10,dimension_encoder_out=3) ->None:
        self.model = AE(input_shape=X.shape[1],dimension_encoder_out=dimension_encoder_out).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                Xbatch = X_t[i:i + batch_size]
                X_pred = self.model(Xbatch)
                loss = self.loss_fn(Xbatch, X_pred)
                self.losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def fit_transform(self,X:np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            X_encoded = self.model(X_t,path="encoder")
        return X_encoded.cpu().numpy()

    def decode(self,X:np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            X_pred = self.model(X_t,path="all")
        return X_pred.cpu().numpy()
