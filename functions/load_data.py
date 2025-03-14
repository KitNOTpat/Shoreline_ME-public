# data select

import pandas as pd
import numpy as np
import os 
from functions.misc import *

class data_select():

    def __init__(self, settings):

        self.settings = settings
        self.target = settings['target']
        self.shoreline = settings['shoreline']

        self.df = pd.read_csv(os.path.join(os.getcwd(),'data','model_input_data', settings['case']+'.csv'), index_col = 'date', parse_dates=True, dayfirst=False)
        self.df.shoreline = self.df.shoreline.interpolate()
        self.df = self.check_inputs()
        
        # Filter the list to include only columns that exist in self.df
        valid_columns = [col for col in settings['dynamic_inputs'] if col in self.df.columns]
        self.inputs = self.df[valid_columns].columns
        self.df = self.df[self.df[self.shoreline].first_valid_index():self.df[self.shoreline].last_valid_index()] 
        

    ##############################
    ##############################
    
    def check_inputs(self):

        Hs_keys = ['Hsig_peak_0','Hsig_peak_1','Hsig_peak_2']
        Tp_keys = ['Tp_peak_0','Tp_peak_1','Tp_peak_2']

        if 'dx' not in self.df.columns:
            self.df['dx'] = self.df[self.shoreline][~self.df[self.shoreline].isna()].diff()

        self.df['E'] = calculate_energy(self.df[Hs_keys])
        
        return self.df[1:]
        
    ##############################
    ##############################
    
    def train_test_split(self):

        first_val_idx = self.df[self.shoreline][:self.settings['sequence_length']+1].index[-1]
        mask = self.df[first_val_idx:].copy()
 
        training_limit = int(0.5 * len(mask))

        self.train = mask[:training_limit].copy()
        self.test = mask[training_limit:].copy()

    ##############################
    ##############################

    def standardize(self):

        dx_m  = self.train[self.settings['target']].mean()
        dx_std = self.train[self.settings['target']].std()
        sl_m = self.train[self.settings['shoreline']].mean()
        sl_std = self.train[self.settings['shoreline']].std()

        self.scalers = ({
            'dx_m': dx_m, 
            'dx_std': dx_std, 
            'shoreline_m': sl_m, 
            'shoreline_std': sl_std
                })
         
        for col in self.df.columns:
            
            temp_mean = self.train[col].mean()
            temp_std = self.train[col].std()
                    
            self.train[col] = (self.train[col] - temp_mean) / temp_std
            self.test[col] = (self.test[col] - temp_mean) / temp_std
            self.df[col] = (self.df[col] - temp_mean) / temp_std

        



    
