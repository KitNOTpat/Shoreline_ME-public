import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from operator import itemgetter

from functions.experts.LSTM.model import LSTM

# Mixture of Experts model
class MixtureOfExperts(nn.Module):
    def __init__(self, n_inputs, settings):
        super(MixtureOfExperts, self).__init__()

        self.threshold = settings['threshold']
        self.settings = settings

        self.lstm_expert = LSTM(n_inputs, settings)
        self.storm_expert = nn.Sequential(
                                nn.Linear(3, 1, bias = True))
        
        with torch.no_grad():  
            self.storm_expert[0].weight.fill_(settings['init_w'])
             
    def forward(self, X, init_hc, varIdx):

        current_input = X[-1][-1]
        Hs_keys = ['Hsig_peak_0','Hsig_peak_1','Hsig_peak_2']
        Tp_keys = ['Tp_peak_0','Tp_peak_1','Tp_peak_2']

        Hs_idx = itemgetter(*Hs_keys)(varIdx)
        Tp_idx = itemgetter(*Tp_keys)(varIdx)

        if (current_input[list(Hs_idx)] > self.threshold).any():
            
            E = torch.reshape(current_input[varIdx['E']], (1,))
            Tp_max = torch.max(current_input[list(Tp_idx)]).reshape(1)
            prev_x = torch.reshape(current_input[-1], (1,))

            storm_input = torch.cat((E, prev_x, Tp_max), axis=0)
            storm_output = self.storm_expert(storm_input)

            # return storm_output, torch.tensor([0]), None
            return storm_output, torch.tensor([0]), init_hc
        
        else: 

            lstm_output, _, init_hc = self.lstm_expert(X, init_hc)
            
        return lstm_output, torch.tensor([1]), init_hc



