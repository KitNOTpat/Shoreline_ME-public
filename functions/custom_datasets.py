# implement custom data loaders for LSTM, linear and non-linear probe

import torch
from torch.utils.data import Dataset

#####################################################
#####################################################

class SequenceDataset(Dataset):
    def __init__(self, dataframe, data, settings):

        self.inputs = data.inputs
        self.target =  settings.get('target')
        self.sequence_length = settings.get('sequence_length')

        self.df = data.df[self.inputs]
        self.start_pos = self.df.index.get_loc(dataframe.index[0])

        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.inputs].values).float()

        self.varIdx= {var: Idx for Idx, var in enumerate(dataframe[self.inputs])}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):

        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
            
        else:
            hist = torch.tensor(self.df[self.start_pos - self.sequence_length+1 + i: self.start_pos+i].values).float()
            current_value = torch.reshape(self.X[i],(1,len(self.inputs)))
            x = torch.cat((hist, current_value), axis = 0)

        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            y = self.y[i_start:(i + 1)]
        else:
            padding = self.y[0].repeat(self.sequence_length - i - 1, 1)
            y = self.y[0:(i + 1)]
            y = torch.cat((padding, torch.reshape(y,(len(y),1))), 0)
        
        return x, self.y[i], i
