import torch
import numpy as np
import matplotlib.pyplot as plt
from c_fid.ts2vec import TS2Vec
import time
from scipy.linalg import sqrtm
import pickle
import mgzip

# adapt from https://github.com/mbohlkeschneider/psa-gan

class MinMaxScaler():

    def __init__(self):
        self.mini = None
        self.range = None

    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data
    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



# create argument parser
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--method_name', type=str, default = None)
parser.add_argument('--dataset_name', type=str, default = None)
parser.add_argument('--dataset_state', type=str, default = None)
parser.add_argument('--gpu_id', type=int, default = None)

args = parser.parse_args()

method_name = args.method_name
dataset_name = args.dataset_name
dataset_state = args.dataset_state
gpu_id = args.gpu_id



with mgzip.open('./data/' + dataset_name + '_' + dataset_state + '.pkl', 'rb') as f:
    ori_data = pickle.load(f)


with mgzip.open('./data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_gen.pkl', 'rb') as f:
    generated_data = pickle.load(f)
generated_data = np.array(generated_data)

print(ori_data.shape, generated_data.shape)
x_train = ori_data
x_gen = generated_data

config = dict(
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000
    )


time_all = []

fid_s = []
for i in range(5):
    start = time.time()
    model = TS2Vec(
        input_dims=x_train.shape[-1],
        device=gpu_id,
        **config
    )
    model.fit(x_train, verbose=True)
    ori_repr = model.encode(x_train, encoding_window='full_series')
    gen_repr = model.encode(x_gen, encoding_window='full_series')
    select = x_gen.shape[0]
    idx = np.random.permutation(select)
    ori = ori_repr[idx]
    gen = gen_repr[idx]
    

    fid_s.append(calculate_fid(ori, gen))
    end = time.time()
    time_all.append(end-start)
    


with open('./data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_fid.pkl', 'wb') as f:
    pickle.dump(np.array(fid_s), f)



# add a use case
# python cal_fid.py --method_name ours/vanilla --dataset_name energy_long --dataset_state train --gpu_id 0
