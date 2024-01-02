import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
from tslearn.datasets import UCR_UEA_datasets
import pickle
import mgzip


class MinMaxScaler():
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


# adapt from https://github.com/TheDatumOrg/VUS
def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    base = 3
    nobs = len(data)
    nlags = int(min(10 * np.log10(nobs), nobs - 1))
    auto_corr = acf(data, nlags=nlags, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        # print(local_max[max_local_max]+base)
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125


# ===============================
# load from csv file
ori_data = np.loadtxt('./DATASET_NAME.csv', delimiter = ",",skiprows = 1)
print(ori_data.shape)
# ===============================
# load from pickle file
with mgzip.open('./DATASET_NAME.pkl', 'rb') as f:
    ori_data = pickle.load(f)
ori_data.shape
# ===============================
# load from existing datasets in UCR/UEA
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("DATASET_NAME")
X_train = X_train.reshape(X_train.shape[1], X_train.shape[0])
ori_data = X_train.copy()
print(ori_data.shape)
for i in range(ori_data.shape[1]):
    ori_data[:,i] = pd.Series(ori_data[:,i]).interpolate().values
# ===============================

window_all = []
for i in range(ori_data.shape[1]):
    window_all.append(find_length(ori_data[:,i]))

seq_len = int(np.mean(np.array(window_all)))
print(seq_len)


# Preprocess the dataset
temp_data = []    
# Cut data by sequence length
for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
    
# Mix the datasets (to make it similar to i.i.d)
idx = np.random.permutation(len(temp_data))    
data = []
for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])


full_train_data = np.array(data)
N, T, D = full_train_data.shape   
print('data shape:', N, T, D) 

valid_perc = 0.1

# further split the training data into train and validation set - same thing done in forecasting task
N_train = int(N * (1 - valid_perc))
N_valid = N - N_train

# Shuffle data
np.random.shuffle(full_train_data)

train_data = full_train_data[:N_train]
valid_data = full_train_data[N_train:]   
print("train/valid shapes: ", train_data.shape, valid_data.shape)    


scaler = MinMaxScaler()        
scaled_train_data = scaler.fit_transform(train_data)
scaled_valid_data = scaler.transform(valid_data)


# Store the preprocessed dataset
dataset_name = 'DATASET_NAME'
with mgzip.open('./data/' + dataset_name + '_train.pkl', 'wb') as f:
    pickle.dump(scaled_train_data, f)

with mgzip.open('./data/' + dataset_name + '_valid.pkl', 'wb') as f:
    pickle.dump(scaled_valid_data, f)

with mgzip.open('./data/' + dataset_name + '_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
