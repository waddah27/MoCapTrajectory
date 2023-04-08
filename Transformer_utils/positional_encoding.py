import torch
from torch import Tensor


def positional_encoding(seq_len:int, dim_model: int, device: torch.device = torch.device('cpu')) -> Tensor:
    '''
    We have to provide positional information to the model, 
    so that it knows about the relative position of data points in the input sequences.
    Mathimatically:
        P(K, 2i) = sin(K/n^(2i/d))
        P(K, 2i+1) = cos(K/n^(2i/d))
        Where:
            K: is the index of the object in the sequence to get the positional encoding of.
            P(K,j) is the function for mapping the Kth obj's pos to idx (K,j) of the positional matrix
            d: dimension of the output embedding space (here we set to 512)
            i: used for mapping to column indices of positional matrix (0<=i<d/2 -> i = [0, ..., d//2 -1])
            n: a User defined scalar set to 1e4 (10000) by paper (Attention Is All You Need)
    :param seq_len: length of the sequence
    :param dim_model: dimentions of the model
    :param device: the device on which the transformer will be trained
    :return Tensor 
    '''

    K = torch.arange(seq_len, dtype=torch.float32, device=device).reshape(1, -1, 1) # K
    j = torch.arange(dim_model, dtype=torch.float32, device=device).reshape(1, 1, -1) #i indices vector
    i = torch.where(j.long()%2==0, j[-1]//2, (j[-1]-1)/2)
    
    phase = K / (1e4 **(2*i / dim_model))
    # dim.long() == indices of the j vector
    return torch.where(j.long() %2 ==0, torch.sin(phase), torch.cos(phase))

if __name__=='__main__':
    # for debugging
    p = positional_encoding(seq_len=4,dim_model=6)
    print(p)
