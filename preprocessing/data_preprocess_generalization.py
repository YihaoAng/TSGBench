import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np

import matplotlib.patches as mpatches 
from tslearn.datasets import UCR_UEA_datasets

import pickle
import mgzip




class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data




usr = 14
dataset_name = 'hapt_' + str(usr)
with mgzip.open('./data/' + dataset_name + '_train.pkl', 'rb') as f:
    ori_data1 = pickle.load(f)

with mgzip.open('./data/' + dataset_name + '_valid.pkl', 'rb') as f:
    ori_data_valid1 = pickle.load(f)

usr = 0
dataset_name = 'hapt_' + str(usr)
with mgzip.open('./data/' + dataset_name + '_train.pkl', 'rb') as f:
    ori_data2 = pickle.load(f)

with mgzip.open('./data/' + dataset_name + '_valid.pkl', 'rb') as f:
    ori_data_valid2 = pickle.load(f)

print(ori_data1.shape, ori_data_valid2.shape)

# ori_data = np.concatenate((ori_data1, ori_data2), axis=0)
# ori_data_valid = np.concatenate((ori_data_valid1, ori_data_valid2), axis=0)

ori_data_valid2 = ori_data_valid2[0:int(ori_data1.shape[0]*0.1),:,:]
print(ori_data_valid2.shape)
ori_data = np.concatenate((ori_data1, ori_data_valid2), axis=0)
np.random.shuffle(ori_data)


dataset_name = 'hapt_cross_' + str(usr1) + '_' + str(usr2)
with mgzip.open('./data/' + dataset_name + '_train.pkl', 'wb') as f:
    pickle.dump(ori_data, f)



