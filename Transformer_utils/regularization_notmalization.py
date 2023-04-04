from torch import nn, Tensor

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, *tensors: Tensor) -> Tensor:
        '''
        Assume that the "query" tensor is given first, so we can compute the
        residual.  This matches the signature of 'MultiHeadAttention'.
        '''
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))