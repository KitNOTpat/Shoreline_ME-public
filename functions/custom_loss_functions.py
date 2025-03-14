# custom_loss_functions

import torch

#####################################################
#####################################################

class cumulative_dx_loss():
    def  __init__(self):
        super(cumulative_dx_loss, self).__init__()

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