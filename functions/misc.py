# miscellaneous functions

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
import numpy as np
# import imageio
from math import pi
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from sklearn.metrics import mean_squared_error, r2_score

#####################################################
#####################################################

def calculate_skill(obs, preds):

    obs_non_nan = obs[~np.isnan(obs)]
    preds_non_nan = preds[~np.isnan(obs)]

    mse = mean_squared_error(obs_non_nan, preds_non_nan)     
    true_variance = np.var(obs_non_nan)
    nmse = mse / true_variance
    
    r2 = r2_score(obs_non_nan, preds_non_nan)

    return nmse, r2

#####################################################
#####################################################

def brier_skill_score(preds, base, obs):

    obs_non_nan = obs[~np.isnan(obs)]
    preds_non_nan = preds[~np.isnan(obs)]
    base_non_nan = base[~np.isnan(obs)]

    mse = mean_squared_error(obs_non_nan, preds_non_nan)   
    base_mse = mean_squared_error(obs_non_nan, base_non_nan) 

    bss = 1 - (mse/base_mse)  
 
    return bss

#####################################################
#####################################################

def calculate_power(H,T):
    Power = ((1025 * (9.81**2) * (H**2) * (T**2))/(8*pi))
    return Power/1e6

#####################################################
#####################################################

def calculate_energy(H):

    delta = H.index.to_series().diff()[1]
    dt = delta.total_seconds() / 3600
    # br 
    E = ((1/16) * 1025 * 9.81 * dt * (H**2).sum(axis=1))/1e6
    # E = ((1/16) * 1025 * 9.81 * dt * (H**2))/1e6

    return E

#####################################################
#####################################################

def destandardize(arr, scalers, target):
    _arr_ = (arr*scalers[f'{target}_std'])+scalers[f'{target}_m']
    return _arr_

#####################################################
#####################################################

def standardize(arr, scalers, target):
    _arr_ = (arr - scalers[f'{target}_m'])/scalers[f'{target}_std']
    return _arr_

#####################################################
#####################################################

def stdize(df):
    std_df = (df - df.mean(axis=0))/df.std()
    return std_df

#####################################################
#####################################################

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

#####################################################
#####################################################

def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

#####################################################
#####################################################

def plot_timeseries(data):

    fig, ax = plt.subplots(figsize=(12,4),facecolor='white')
    fig.tight_layout(pad=5.0)
    ax.grid(linestyle = '--', linewidth = 1, axis='both')

    ax.plot(data.index, data, c = 'mediumvioletred')

#####################################################
#####################################################

def plot_train_test(data, settings):
    fig, ax = plt.subplots(figsize=(12,4),facecolor='white')
    fig.tight_layout(pad=5.0)
    ax.grid(linestyle = '--', linewidth = 1, axis='both')
    ax.scatter(data.train.index, data.train[settings['shoreline']], c = 'royalblue', s = 10)
    ax.scatter(data.test.index, data.test[settings['shoreline']], c = 'crimson', s = 10)
    ax.scatter(data.df.index, data.df[settings['shoreline']], c = 'grey', s = 10, zorder=0, alpha = 0.5)

#####################################################
#####################################################

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = 23, 2
        
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch
    

##############################################

# Custom styling function
def highlight_metrics(df):
    # A mask for highlighting, initialized with False
    highlight_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    # Loop through columns and apply custom highlighting based on metric
    for col in df.columns:
        if col[1] == "NMSE":
            # For NMSE, highlight the minimum value in each row
            is_min = df[col] == df.xs('NMSE', axis=1, level=1).min(axis=1)
            highlight_mask[col] = is_min
        elif col[1] == "R2":
            # For R2, highlight the maximum value in each row
            is_max = df[col] == df.xs('R2', axis=1, level=1).max(axis=1)
            highlight_mask[col] = is_max
    
    # Apply color based on the mask
    return highlight_mask.replace({True: 'background-color: green', False: ''})