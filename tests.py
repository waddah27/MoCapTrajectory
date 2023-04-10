import matplotlib.pyplot  as plt
from TimeSeriesTransformer.TimeSeriesTransformer import TimeSeriesTransformer
from Transformer_utils.positional_encoding import PositionalEncoder
import torch

if __name__=='__main__':
    
    src = torch.rand(64, 32, 512)
    tgt = torch.rand(64, 16, 512)
    out = TimeSeriesTransformer()(src, tgt)
    print(out.shape)
    pe = PositionalEncoder(seq_len=100, dim_model=512)(torch.Tensor([0]))
    cax = plt.matshow(pe[0])
    plt.gcf().colorbar(cax)
    plt.show()
