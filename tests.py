import os
import torch
import matplotlib.pyplot  as plt
import torch.nn as nn
from TimeSeriesTransformer.Transformer import Transformer
from Transformer_utils.positional_encoding import PositionalEncoder
from PrepareData.dataset import TransformerDataset
from TimeSeriesTransformer.TSTransformer import TimeSeriesTransformer, model_parameters


from read_mocap import PATH_TO_DATA, PATH_TO_RECORDED_DATA, read_and_preprocess_csv_data

if __name__=='__main__':
    mp = model_parameters()
    # Test reading dataset from costum class
    print(f'path to recorded data = {PATH_TO_RECORDED_DATA}')
    # Get the csv records only
    records = [f for f in sorted(os.listdir(PATH_TO_RECORDED_DATA)) if f.endswith('.csv')]
    print(f'recorded list = {records}')
    Data = TransformerDataset(os.path.join(PATH_TO_RECORDED_DATA,records[-1]), src_target_split=0.7, start_col=10)
    src, trg, trg_y = Data.get_src_trg()
    src = torch.tensor(src.iloc[:,:].values)
    trg = torch.tensor(trg.iloc[:,:].values)
    
    TSmodel = TimeSeriesTransformer(
        dim_val=mp.dim_val,
    input_size=mp.input_size, 
    dec_seq_len=mp.dec_seq_len,
    max_seq_len=mp.max_seq_len,
    out_seq_len=mp.output_sequence_length, 
    n_decoder_layers=mp.n_decoder_layers,
    n_encoder_layers=mp.n_encoder_layers,
    n_heads=mp.n_heads
    )
    
    out = TSmodel(
        src=src,
        tgt = trg
    )
    layer = torch.nn.Linear(9, 512)
    norm = nn.LayerNorm(512)
    Dropout = nn.Dropout()
    # layer2 = torch.nn.Linear(512, 2024)
    
    src = torch.rand(64, 32, 9)
    tgt = torch.rand(64, 16, 9)
    normed = src[0]
    droped = Dropout(layer(src))
    # b = norm(src[0] + Dropout(layer(src)))
    model =Transformer()
    out = model(src, tgt)
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
