import torch
from torch import Tensor


def positional_encoding(seq_len:int, dim_model: int, device: torch.device = torch.device('cpu')) -> Tensor:
    '''
    We have to provide positional information to the model, 
    so that it knows about the relative position of data points in the input sequences.
    :param seq_len: length of the sequence
    :param dim_model: dimentions of the model
    :param device: the device on which the transformer will be trained
    :return Tensor 
    '''
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float32, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 **(dim / dim_model))
    return torch.where(dim.long() %2 ==0, torch.sin(phase), torch.cos(phase))
