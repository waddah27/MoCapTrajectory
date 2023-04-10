import torch
import matplotlib.pyplot  as plt
import torch.nn as nn
from TimeSeriesTransformer.TimeSeriesTransformer import TimeSeriesTransformer
from Transformer_utils.positional_encoding import PositionalEncoder


if __name__=='__main__':
    
    src = torch.rand(64, 32, 512)
    tgt = torch.rand(64, 16, 512)
    out = TimeSeriesTransformer()(src, tgt)
    print(out.shape)
    pe = PositionalEncoder(seq_len=100, dim_model=512)(torch.LongTensor([0]))
    print(f'pe shape = {pe.shape}')
    # To use positional encoding matrix as weights for embedding layer:
    embedding_layer = nn.Embedding.from_pretrained(pe[0])
    # To Freeze the weights from changing during backpropagation:
    embedding_layer.weight.requiresGrad = False
    embedded_output = embedding_layer((torch.LongTensor([0])))
    print(f'embedding_layer output shape = {embedded_output.shape}')
    cax = plt.matshow(pe[0])
    plt.gcf().colorbar(cax)
    plt.show()
