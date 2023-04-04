import torch
from Transformer_utils.feed_forward import feed_forward
from Transformer_utils.MultiHeadAttention import MultiHeadAttention
from Transformer_utils.positional_encoding import positional_encoding
from Transformer_utils.regularization_notmalization import Residual
from Transformer_utils.common_utils import CFG
from torch import nn, Tensor

cfg = CFG()
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model: int = cfg.dim_model,
                 n_heads:int = cfg.n_heads,
                 dim_feedforward: int = cfg.dim_feedforward, 
                 dropout:float = cfg.dropout,
                 ) -> None:
        super().__init__()
        dim_q = dim_k = max(dim_model // n_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(
            num_heads = n_heads,
            dim_in = dim_model,
            dim_q = dim_q,
            dim_k = dim_k
            ),
            dim = dim_model,
            dropout = dropout, 
        )
        self.feed_forward = Residual(
            feed_forward(dim_input=dim_model, dim_feedforward=dim_feedforward),
            dim = dim_model,
            dropout = dropout,
        )
    
    def forward(self, src:Tensor)->Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 n_layers: int = cfg.n_decoder_layers,
                 dim_model: int = cfg.dim_model,
                 n_heads: int = cfg.n_heads,
                 dim_feedforward: int = cfg.dim_feedforward,
                 dropout: float = cfg.dropout, 
                 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
            TransformerEncoderLayer(dim_model=dim_model, 
                                    n_heads=n_heads, 
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout
                                    )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src:Tensor)-> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src+=positional_encoding(seq_len=seq_len, dim_model=dimension)
        for layer in self.layers:
            src = layer(src)
        return src
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim_model: int = cfg.dim_model,
                 n_heads:int = cfg.n_heads,
                 dim_feedforward: int = cfg.dim_feedforward,
                 dropout:float = cfg.dropout,
                 ) -> None:
        super().__init__()
        dim_q = dim_k = max(dim_model//n_heads, 1)
        self.attention1 = Residual(
            MultiHeadAttention(
            num_heads=n_heads,
            dim_in=dim_model,
            dim_q=dim_q,
            dim_k=dim_k,
            ),
            dim=dim_model,
            dropout=dropout,
        )
        self.attention2 = Residual(
            MultiHeadAttention(
            num_heads=n_heads,
            dim_in=dim_model,
            dim_q=dim_q,
            dim_k=dim_k,
            ),
            dim=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(
            dim_input=dim_model,
            dim_feedforward=dim_feedforward,
            ),
            dim=dim_model,
            dropout=dropout,
        )
    def forward(self, target: Tensor, memory: Tensor)-> Tensor:
        target = self.attention1(target, target, target)
        target = self.attention2(target, memory, memory)
        return self.feed_forward(target)

class TransformerDecoder(nn.Module):
    def __init__(self,
                 n_layers:int = cfg.n_decoder_layers,
                 dim_model: int = cfg.dim_model,
                 n_heads:int = cfg.n_heads,
                 dim_feedforward: int = cfg.dim_feedforward,
                 dropout:float = cfg.dropout,
                 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
            TransformerDecoderLayer(dim_model=dim_model,
                                    n_heads=n_heads,
                                    dim_feedforward=dim_feedforward,
                                    )
                            for _ in range(n_layers)
            ]
        )

        self.Linear = nn.Linear(dim_model, dim_model)
    def forward(self, target:Tensor, memory:Tensor)->Tensor:
        seq_len, dimension = target.size(1), target.size(2)
        target+=positional_encoding(seq_len=seq_len, dim_model=dimension)
        for layer in self.layers:
            target = layer(target, memory)
        return torch.softmax(self.Linear(target), dim=-1)
    


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 n_encoder_layers:int = cfg.n_encoder_layers,
                 n_decoder_layers: int = cfg.n_decoder_layers,
                 dim_model:int = cfg.dim_model,
                 n_heads:int = cfg.n_heads,
                 dim_feedforward:int = cfg.dim_feedforward,
                 dropout:float = cfg.dropout,) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            n_layers=n_encoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            n_layers=n_decoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    
    def forward(self, src:Tensor, target:Tensor)->Tensor:
        return self.decoder(target, self.encoder(src))

