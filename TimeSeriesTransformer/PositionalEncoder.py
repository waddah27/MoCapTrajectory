import torch
import math
from torch import nn, Tensor

class PositionalEncoder(nn.Module):
    def __init__(self,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000,
                 d_model: int = 512,
                 batch_first: bool = False) -> None:
        """
        parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: the dimension fo the output of the sub-layers in the model
            (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
    