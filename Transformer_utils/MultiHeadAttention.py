import torch
from torch import Tensor
import torch.nn.functional as f
from torch import nn


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    '''
    Q, K, and V (query, key, and value arrays) are batches of matrices, 
    each with shape (batch_size, sequence_length, num_features). 
    Batch matrix multiplication is only performed over the last two dimensions.
    '''
    attention = query.bmm(key.transpose(1,2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(attention / scale, dim=-1)
    return softmax.bmm(value)

class HeadAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int) -> None:
        super().__init__()
        
        self.heads = nn.ModuleList([HeadAttention(dim_in, dim_q, dim_k) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * dim_k, dim_in)
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(torch.cat([h(query, key, value) for h in self.heads], dim=-1))
