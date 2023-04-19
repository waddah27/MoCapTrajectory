import torch
from typing import Tuple
import pandas as pd

class TransformerDataset:
    def __init__(self, path: str, src_target_split:float, start_col:int=0, end_col:int=None) -> None:
        self.path = path
        self.start_col = start_col
        self.end_col = end_col 
        self.src_target_split = src_target_split
        self.data = pd.read_csv(self.path)
    def read_and_preprocess_csv_data(self):
        '''
        This Function reads the data from csv file,
        convert strings to numeric and does forward
        and backward linear interpolation to fill the NaNs
            :param path: the path to csv file
            :param start_col: idx of the data's 1st col needed (to take a slice)
            :param end_col: idx of the data's final col needed (to take a slice)
            :return data: the csv dataframe after doing the abovementioned preprocessing
            '''
        
        if self.end_col is not None:
            self.data = self.data.iloc[:, self.start_col:self.end_col] # rotation around (X, Y, Z) + Postition (X, Y, Z)
        else:
            self.data = self.data.iloc[:, self.start_col:]
        # Convert strings to numeric
        for col in self.data:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Used forward and backward linear interpolation for filling nan values
        self.data.interpolate(method='linear', limit_direction='both', inplace=True)
        return self.data
    def get_src_trg(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence. 
        Args:
            self.data: tensor, a 3D tensor of length n where 
                    n = encoder input length + target sequence length  
            src_target_split: int, the desired length of the source and target input to the transformer encoder
        Return: 
            src: tensor, 3D, used as input to the transformer model
            trg: tensor, 3D, used as input to the transformer model
            trg_y: tensor, 3D, the target sequence against which the model output
                is compared when computing loss. 
        
        """
        # assert self.data.shape[0] == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"
        # encoder input
        self.data = self.read_and_preprocess_csv_data()
        thresh = int(self.src_target_split*self.data.shape[0])
        src = self.data.iloc[:thresh, :] 
        
        # decoder input. It must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = self.data.iloc[thresh:,:-1]

        # trg = trg[:, 0]

        #print("From data.TransformerDataset.get_src_trg: trg shape after slice: {}".format(trg.shape))

        if len(trg.shape) == 1:

            trg = trg.unsqueeze(-1)

            #print("From data.TransformerDataset.get_src_trg: trg shape after unsqueeze: {}".format(trg.shape))

        
        # assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = self.data.iloc[-trg.shape[0]:]

        #print("From data.TransformerDataset.get_src_trg: trg_y shape before slice: {}".format(trg_y.shape))

        # We only want trg_y to consist of the target variable not any potential exogenous variables
        # trg_y = trg_y[:, 0] #???

        #print("From data.TransformerDataset.get_src_trg: trg_y shape after slice: {}".format(trg_y.shape))

        assert trg_y.shape[0] == trg.shape[0], "Length of trg_y does not match target sequence length"

        return src, trg, trg_y#.squeeze(-1) # ??? change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] 

    