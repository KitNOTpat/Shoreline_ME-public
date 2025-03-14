import numpy as np
import pandas as pd
import torch

from functions.misc import *

#####################################################
#####################################################

def train(df, df_train, data_loader, model, loss_function, scalers, optimizer, settings, DEVICE):

    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for ii, (X, y, I) in enumerate(data_loader):

        X, y = X.to(DEVICE), y.to(DEVICE)

        batch_size = X.shape[0]
        seq_len = X.shape[1]
        init_position = I[0].item()
        init_hc = None
        preds = torch.rand(batch_size)

        for idx, thisX in enumerate(X):
            
            if idx == 0:  

                init_position_idx = df.index.get_loc(df_train.index[init_position])
                prev_x = init_prevX(df, seq_len, init_position_idx, settings).to(DEVICE)

            else:

                prev_x = torch.cat((prev_x[1:].flatten(), shoreline_in.detach()), axis = 0)
                prev_x = torch.reshape(prev_x, (seq_len,1))
                prev_x = standardize(prev_x, scalers, 'shoreline')

            thisX = torch.cat((thisX,prev_x),axis = 1)
            thisX = thisX.unsqueeze(dim=0).to(DEVICE)
            
            thispred, _, init_hc = model(thisX, init_hc, scalers['varIdx'])

            thispred = destandardize(thispred, scalers, 'dx').to(DEVICE)
            prev_x = destandardize(prev_x, scalers, 'shoreline').to(DEVICE)
                
            preds[idx] = thispred

            shoreline_in = torch.add(prev_x.flatten()[-1], thispred).to(DEVICE)


        y = destandardize(y, scalers, settings['target']).to(DEVICE)
        loss = loss_function.forward(preds, y)

        # ----> L1 Regularization <----
        lambda_ = settings['lambda']
        l1_loss = torch.tensor(0., device = DEVICE)
        for param in model.parameters():
            l1_loss = l1_loss +  torch.norm(param, 1)
        loss = loss + lambda_ * l1_loss 

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
     
    avg_loss = total_loss / num_batches
    return avg_loss, preds, y


#####################################################
#####################################################

def predict(df_train, data_loader, model, scalers, start_date, settings):

    output = torch.tensor([])
    weight = []
    init_hc = None
    model.eval()
    
    with torch.no_grad():
         
        for X, _, I in data_loader:
            
            batch_size = X.shape[0]
            seq_len = X.shape[1]
            init_position = I[0].item()
             
            batch_preds = torch.zeros(batch_size)

            for idx, thisX in enumerate(X):
                
                if init_position == 0:
                    prev_x = init_prevX(df_train, seq_len, start_date, settings)

                else:

                    prev_x = torch.cat((prev_x[1:].flatten(), shoreline_in.detach()), axis = 0)
                    prev_x = torch.reshape(prev_x, (seq_len,1))
                    prev_x = standardize(prev_x, scalers, 'shoreline')

                thisX = torch.cat((thisX,prev_x),axis = 1)
                thisX = thisX.unsqueeze(dim=0)

                thispred, P, init_hc = model(thisX, init_hc, scalers['varIdx'])

                weight.append(P.detach().numpy().flatten())
                
                thispred = destandardize(thispred, scalers, 'dx')
                prev_x = destandardize(prev_x, scalers, 'shoreline')  

                batch_preds[idx] = torch.add(prev_x.flatten()[-1], thispred)
                shoreline_in = torch.add(prev_x.flatten()[-1], thispred)
    

            output = torch.cat((output,batch_preds.flatten()[-batch_size:]), 0)
   
    return output, pd.DataFrame(weight)

#####################################################
#####################################################

def init_prevX(df, seq_len, position, settings):

    if seq_len > 1:
        
        prev_sl_position = df[settings['shoreline']][:position][-seq_len:]
        
        if prev_sl_position.isna().all():
            last_valid_idx = df[settings['shoreline']][:position].last_valid_index()
            last_valid_sl = df[settings['shoreline']][last_valid_idx]
            prev_sl_position.iloc[:] = last_valid_sl

        elif prev_sl_position.isna().any():
            prev_sl_position = prev_sl_position.interpolate(limit_direction='both')
        
        prev_x = torch.tensor(prev_sl_position).float().reshape(seq_len,1)

    else:

        last_valid_idx = df[settings['shoreline']][:position].last_valid_index()
        last_sl_position = df[settings['shoreline']][last_valid_idx]
        prev_x = torch.tensor(last_sl_position).float().reshape(seq_len,1)

    return prev_x