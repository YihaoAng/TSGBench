import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
from .ds_ps import discriminative_score_metrics, predictive_score_metrics
from .ts2vec import initialize_ts2vec
from .feature_based_measures import calculate_mdd, calculate_acd, calculate_sd, calculate_kd
from .visualization import visualize_tsne, visualize_distribution
from .utils import show_with_start_divider, show_with_end_divider, determine_device, write_json_data


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

def calculate_ed(ori_data,gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0
        for j in range(n_series):
            distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_distance_eu += distance
        distance_eu.append(total_distance_eu / n_series)

    distance_eu = np.array(distance_eu)
    average_distance_eu = distance_eu.mean()
    return average_distance_eu

def calculate_dtw(ori_data,comp_data):
    distance_dtw = []
    n_samples = ori_data.shape[0]
    for i in range(n_samples):
        distance = multi_dtw_distance(ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True)
        distance_dtw.append(distance)

    distance_dtw = np.array(distance_dtw)
    average_distance_dtw = distance_dtw.mean()
    return average_distance_dtw


def evaluate_data(cfg, ori_data, gen_data):
    show_with_start_divider(f"Evalution with settings:{cfg}")

    # Parse configs
    method_list = cfg.get('method_list','[C-FID,MDD,ACD,SD,KD,ED,DTW,t-SNE,Distribution]')
    #result_path = cfg.get('result_path',r'./result/')
    dataset_name = cfg.get('dataset_name','dataset')
    model_name = cfg.get('model','TimeVAE')
    no_cuda = cfg.get('no_cuda',False)
    cuda_device = cfg.get('cuda_device',0)
    device = determine_device(no_cuda,cuda_device)
    result_path = cfg.get('result_path','./result/')

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M%S")
    combined_name = f'{model_name}_{dataset_name}_{formatted_time}'

    if not isinstance(method_list,list):
        method_list = method_list.strip('[]')
        method_list = [method.strip() for method in method_list.split(',')]

    # Check original data
    if ori_data is None:
        show_with_end_divider('Error: Original data not found.')
        return None
    if isinstance(ori_data, (list, tuple)) and len(ori_data) == 2:
        train_data, valid_data = ori_data
        ori_data = train_data
    else:
        show_with_end_divider('Error: Original data is invalid.')
        return None

    # Check original data
    if gen_data is None:
        show_with_end_divider('Error: Generated data not found.')
        return None
    if ori_data.shape != gen_data.shape:
        print(f'Original data shape: {ori_data.shape}, Generated data shape: {gen_data.shape}.')
        show_with_end_divider('Error: Generated data does not have the same shape with original data.')
        return None
    
    # Execute eval method in method list
    result = {}

    # Model-based measures
    if 'DS' in method_list:
        iter_disc = cfg.get('iter_disc',2000)
        rnn_name = cfg.get('rnn_name','lstm')
        disc = discriminative_score_metrics(ori_data, gen_data, iterations = iter_disc, rnn_name = rnn_name)
        result['DS'] = disc
    if 'PS' in method_list:
        iter_pred = cfg.get('iter_disc',2000)
        rnn_name = cfg.get('rnn_name','lstm')
        pred = predictive_score_metrics(ori_data, gen_data, iterations = iter_pred, rnn_name = rnn_name)
        result['PS'] = pred
    if 'C-FID' in method_list:
        fid_model = initialize_ts2vec(np.transpose(train_data, (0, 2, 1)),device)
        ori_repr = fid_model.encode(np.transpose(ori_data,(0, 2, 1)), encoding_window='full_series')
        gen_repr = fid_model.encode(np.transpose(gen_data,(0, 2, 1)), encoding_window='full_series')
        cfid = calculate_fid(ori_repr,gen_repr)
        result['C-FID'] = cfid

    # Feature-based measures
    if 'MDD' in method_list:
        mdd = calculate_mdd(ori_data,gen_data)
        result['MDD'] = mdd
    if 'ACD' in method_list:
        acd = calculate_acd(ori_data,gen_data)
        result['ACD'] = acd
    if 'SD' in method_list:
        sd = calculate_sd(ori_data,gen_data)
        result['SD'] = sd
    if 'KD' in method_list:
        kd = calculate_kd(ori_data,gen_data)
        result['KD'] = kd
    
    # Distance-based measures
    if 'ED' in method_list:
        ed = calculate_ed(ori_data,gen_data)
        result['ED'] = ed
    if 'DTW' in method_list:
        dtw = calculate_dtw(ori_data,gen_data)
        result['DTW'] = dtw

    # Visualization
    if 't-SNE' in method_list:
        visualize_tsne(ori_data, gen_data, result_path, combined_name)
    if 'Distribution' in method_list:
        visualize_distribution(ori_data, gen_data, result_path, combined_name)
    #print(f'Evaluation results:{result}.')

    if isinstance(result, dict):
        result_path = os.path.join(result_path, f'{combined_name}.json')
        write_json_data(result, result_path)
        print(f'Evaluation results saved to {result_path}.')
    
    show_with_end_divider(f'Evaluation done. Results:{result}.')

    return result