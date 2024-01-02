import os 
import numpy as np
import ml_collections
import yaml
import torch
from torch import nn
from os import path as pt
from typing import List, Tuple
import mgzip 
import pickle
from dtaidistance.dtw_ndim import distance as multi_dtw_distance

# ====================================
# adapt from https://github.com/DeepIntoStreams/Evaluation-of-Time-Series-Generative-Models
def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    for i in range(D):
        # Compute the correlation between X_{t, d} and X_{t-tau, d}
        if hasattr(torch,'corrcoef'):
            correlations[:, :, i] = torch.corrcoef(X[:, :, i].t())
        else: 
            correlations[:, :, i] = torch.from_numpy(np.corrcoef(to_numpy(X[:, :, i]).T))

    if not symmetric:
        # Loop through each time step from lag to T-1
        for t in range(T):
            # Loop through each lag from 1 to lag
            for tau in range(t+1, T):
                correlations[tau, t, :] = 0

    return correlations



def cacf_torch(x, lags: list, dim=(0, 1)):
    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))

def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b+1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins+1)
    delta = bins[1]-bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins



def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew



def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis






class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

    
class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(x_real)

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(x_fake)
        return self.norm_foo(skew_fake - self.skew_real)


class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(x_real)

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(x_fake)
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)

# ====================================

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--method_name', type=str, default = None)
parser.add_argument('--dataset_name', type=str, default = None)
parser.add_argument('--dataset_state', type=str, default = None)
parser.add_argument('--gpu_id', type=str, default = None)

args = parser.parse_args()

method_name = args.method_name
dataset_name = args.dataset_name
dataset_state = args.dataset_state
gpu_id = args.gpu_id

os.environ["CUDA_AVAILABLE_DEVICES"] = gpu_id



with mgzip.open('./data/' + dataset_name + '_' + dataset_state + '.pkl', 'rb') as f:
    ori_data = pickle.load(f)


with mgzip.open('./data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_gen.pkl', 'rb') as f:
    generated_data = pickle.load(f)
generated_data = np.array(generated_data)

print(ori_data.shape, generated_data.shape)
x_real = torch.Tensor(ori_data)
x_fake = torch.Tensor(generated_data)

mdd_all = []
acd_all = []



for i in range(0,5):

    mdd = (HistoLoss(x_real[:, 1:, :], n_bins=50, name='marginal_distribution')(x_fake[:, 1:, :])).detach().cpu().numpy()
    acd = (ACFLoss(x_real, name='auto_correlation', stationary = True)(x_fake)).detach().cpu().numpy()
    
    mdd_all.append(mdd)
    acd_all.append(acd)


mdd_all = np.array(mdd_all)
acd_all = np.array(acd_all)

skewness = SkewnessLoss(x_real = x_real, name='skew')
sd = skewness.compute(x_fake).mean()
sd = float(sd.numpy())

kurtosis = KurtosisLoss(x_real = x_real, name='kurtosis')
kd = kurtosis.compute(x_fake).mean()
kd = float(kd.numpy())

# ====================================

n_samples = ori_data.shape[0]
n_series = ori_data.shape[2]
total_distance_eu = 0
distance_eu = []
for i in range(n_samples):
    total_distance_eu = 0
    ori_sample = ori_data[i]
    generated_sample = generated_data[i]
    for j in range(n_series):
        distance = np.linalg.norm(ori_sample[:, j] - generated_sample[:, j])
        total_distance_eu += distance
    total_distance_eu = total_distance_eu / n_series
    distance_eu.append(total_distance_eu)
    
distance_eu = np.array(distance_eu)
average_distance_eu = distance_eu.mean()

distance_dtw = []
for i in range(n_samples):
    total_distance = 0
    ori_sample = ori_data[i]
    generated_sample = generated_data[i]
    distance = multi_dtw_distance(ori_sample.astype(np.double), generated_sample.astype(np.double), use_c = True)
    total_distance = distance
    distance_dtw.append(total_distance)
    
distance_dtw = np.array(distance_dtw)
average_distance_dtw = distance_dtw.mean()


with open('../data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_eval_feature.pkl', 'wb') as f:
    pickle.dump([mdd_all, acd_all, sd, kd], f)


with open('../data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_eval_distance.pkl', 'wb') as f:
    pickle.dump([average_distance_eu, average_distance_dtw], f)


