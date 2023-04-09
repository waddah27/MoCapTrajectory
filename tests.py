from TimeSeriesTransformer.TimeSeriesTransformer import TimeSeriesTransformer
import torch
if __name__=='__main__':
    
    src = torch.rand(64, 32, 512)
    tgt = torch.rand(64, 16, 512)
    out = TimeSeriesTransformer()(src, tgt)
    print(out.shape)
