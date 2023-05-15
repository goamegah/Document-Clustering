from torch import nn
import torch
class AE(nn.Module):
    def __init__(self,dimension_encoder_out=3 ,**kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_hidden_layer1 = nn.Linear(
            in_features=128, out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=dimension_encoder_out
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=dimension_encoder_out, out_features=128
        )
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )
    def forward_encoder(self,x):
        activation = self.encoder_hidden_layer(x)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer1(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code

    def forward_decoder(self,x):
        activation = self.decoder_hidden_layer(x)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer1(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def forward(self, x,path="all"):
        if path=='all':
            x = self.forward_encoder(x)
            x = self.forward_decoder(x)
        elif path=='encoder':
            x = self.forward_encoder(x)
        elif path=='decoder':
            x = self.forward_decoder(x)
        else:
            raise NotImplementedError
        return x
