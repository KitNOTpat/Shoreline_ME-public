 # model defenition

import torch
import torch.nn as nn
import numpy as np

#####################################################
#####################################################

class LSTM(nn.Module):

    '''
    Stacked LSTM model, dictated by input parameters 
    '''

    def __init__(self, n_inputs, settings):
        super(LSTM, self).__init__()

        # print(f"{'Implicit' if implicit else 'Explicit'} Model Type! \n")

        self.n_inputs = n_inputs+1
        self.hidden_units = settings['num_hidden_units']
        self.num_layers = settings['num_lstm_layers']
        self.dropout = settings['neuron_dropout']
        # self.stateful = settings['Stateful']
        self.batch_size = settings['batch_size']

        self.lstm = nn.LSTM(
            input_size = self.n_inputs,
            hidden_size = self.hidden_units,
            batch_first = True,
            num_layers = self.num_layers)

        self.dropout = nn.Dropout(p=self.dropout)
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)
        
    def forward(self, x, init_hc=None, varIdx = None):

        device = next(self.parameters()).device
        
        h0 = torch.zeros(self.num_layers, 1, self.hidden_units, device=device)  # Initialize h0
        c0 = torch.zeros(self.num_layers, 1, self.hidden_units, device=device)  # Initialize c0 

        if init_hc is None:
            _, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            hn, cn = init_hc 
            _, (hn, cn) = self.lstm(x, (hn, cn))

        hn_out = self.dropout(hn[-1])
        out = self.linear(hn_out).flatten()

        return out, torch.tensor([1]), (hn, cn)



#####################################################
#####################################################

class CustomMSE():
    def  __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, output, target):

        if torch.isnan(target).all():
            return torch.tensor([0.0]).requires_grad_()

        n = target.shape[0]
        cum_y = 0
        MSE = 0
 
        output_ = torch.cumsum(output, axis=0)

        for ii, y in enumerate(target):

            if torch.isnan(y):
                continue
            else:
                cum_y = torch.add(y,cum_y)
                target[ii] = cum_y

                err = torch.square(target[ii]-output_[ii])
                MSE = MSE + err

        MSE = (MSE/n)

        return MSE