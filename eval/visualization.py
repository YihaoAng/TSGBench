import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
import mgzip
from sklearn.manifold import TSNE
import argparse

np.random.seed(seed=42)

# adapt from https://github.com/jsyoon0823/TimeGAN, https://openreview.net/forum?id=ez6VHWvuXEx

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



def visualization_tsne(method_name, dataset_name, dataset_state, ax):    
    with mgzip.open('./data/' + dataset_name + '_' + dataset_state + '.pkl', 'rb') as f:
        ori_data = pickle.load(f)
    ori_data = np.array(ori_data)
    with mgzip.open('./data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_gen.pkl', 'rb') as f:
            generated_data = pickle.load(f)

        generated_data = np.array(generated_data)

    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    # anal_sample_no = len(ori_data)
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
    colors = ["C0" for i in range(anal_sample_no)] + ["C1" for i in range(anal_sample_no)]    
    
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    tsne = TSNE(n_components = 2, verbose = 0, perplexity = 30, n_iter = 1000, random_state = 42) # 40, 300
    tsne_results = tsne.fit_transform(prep_data_final)
    
    ax.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.5, label = "Original", s = 5)
    ax.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.5, label = "Generated", s = 5)


    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    


def visualization_distr(method_name, dataset_name, dataset_state, ax):
    
    with mgzip.open('./data/' + dataset_name + '_' + dataset_state + '.pkl', 'rb') as f:
        ori_data = pickle.load(f)
    ori_data = np.array(ori_data)

    with mgzip.open('./data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_gen.pkl', 'rb') as f:
        generated_data = pickle.load(f)

    generated_data = np.array(generated_data)

    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    

    sns.distplot(prep_data, hist = False, kde = True,kde_kws = {'linewidth': 2},label = 'Original', 
                 color = 'C0')
    sns.distplot(prep_data_hat, hist = False, kde = True,kde_kws = {'linewidth': 2,'linestyle':'--'},label = 'Generated',
                color = 'C1')




parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--method_name', type=str, default = None)
parser.add_argument('--dataset_name', type=str, default = None)
parser.add_argument('--dataset_state', type=str, default = None)

args = parser.parse_args()


method_name = args.method_name
dataset_name = args.dataset_name
dataset_state = args.dataset_state


fig, ax = plt.subplots(1,1,figsize = (2,2))
visualization_tsne(method_name, dataset_name, dataset_state, ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
for pos in ['top', 'bottom', 'left', 'right']:
	ax.spines[pos].set_visible(False)
plt.savefig('./figures/' + method_name + '_' + dataset_name + '_tsne.png', dpi=400, bbox_inches='tight')


fig, ax = plt.subplots(1,1,figsize = (2,2))
visualization_distr(method_name, dataset_name, dataset_state, ax = ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xlim(0,1)
for pos in ['top','right']:
	ax.spines[pos].set_visible(False)
plt.savefig('./figures/' + method_name + '_' + dataset_name + '_distr.png', dpi=400, bbox_inches='tight')


