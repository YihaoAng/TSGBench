# TSGBench: Time Series Generation Benchmark

<!-- ![TSG Method Ranking across Ten evaluation Measures and Ten Datasets](https://github.com/YihaoAng/TSGBench/blob/main/figures/ranking.png) -->
<img src="https://github.com/YihaoAng/TSGBench/blob/main/figures/ranking.png" alt="drawing" width="400"/>


- [Introduction](#introduction)
- [TSG Methods](#tsg-methods)
- [Datasets](#datasets)
- [Evaluation Measures](#evaluation-measures)
- [Benchmarking Results](#benchmarking-results)
  - [Main Results](#main-results)
  - [Visualization](#visualization)
  - [Generalization Test](#generalization-test)
- [Get Started with TSGBench](#get-started-with-tsgbench)
   - [Configuration](#configuration)
   - [Run TSGBench](#run-tsgbench)


## Introduction

TSGBench is an open-sourced benchmark for the Time Series Generation task.

![Overall Architecture of TSGBench](https://github.com/YihaoAng/TSGBench/blob/main/figures/overall_architecture.png)




## TSG Methods

TSGBench surveys Time Series Generation (TSG) methods by different backbone models and their specialties.



| Time | Paper        | Model     | Specialty                     | Reference                                                    |
| ---- | ------------ | --------- | ----------------------------- | ------------------------------------------------------------ |
| 2016 | C-RNN-GAN    | GAN       | music                         | https://github.com/olofmogren/c-rnn-gan                      |
| 2017 | RGAN         | GAN       | general (w/ medical) TS       | https://github.com/ratschlab/RGAN                            |
| 2018 | T-CGAN       | GAN       | irregular TS                  | https://github.com/gioramponi/GAN_Time_Series                |
| 2019 | WaveGAN      | GAN       | audio                         | https://github.com/chrisdonahue/wavegan                      |
| 2019 | TimeGAN      | GAN       | general TS                    | https://github.com/jsyoon0823/TimeGAN                        |
| 2020 | TSGAN        | GAN       | general TS                    | Community implementation: https://github.com/Yashkataria/CGAN-for-time-series |
| 2020 | DoppelGANger | GAN       | general TS                    | https://github.com/fjxmlzn/DoppelGANger                      |
| 2020 | SigCWGAN     | GAN       | long financial TS             | https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs |
| 2020 | Quant GANs   | GAN       | long financial TS             | Community implementations: https://github.com/ICascha/QuantGANs-replication, https://github.com/JamesSullivan/temporalCN |
| 2020 | COT-GAN      | GAN       | TS and video                  | https://github.com/tianlinxu312/cot-gan                      |
| 2021 | Sig-WGAN     | GAN       | financial TS                  | https://github.com/SigCGANs/Sig-Wasserstein-GANs             |
| 2021 | TimeGCI      | GAN       | general TS                    | No code available                                            |
| 2021 | RTSGAN       | GAN       | general (w/ incomplete) TS    | https://github.com/acphile/RTSGAN                            |
| 2022 | PSA-GAN      | GAN       | general (w/ forecasting) TS   | https://github.com/mbohlkeschneider/psa-gan                  |
| 2022 | CEGEN        | GAN       | general TS                    | No code available                                            |
| 2022 | TTS-GAN      | GAN       | general TS                    | https://github.com/imics-lab/tts-gan                         |
| 2022 | TsT-GAN      | GAN       | general TS                    | No code available                                            |
| 2022 | COSCI-GAN    | GAN       | general TS                    | https://github.com/aliseyfi75/COSCI-GAN                      |
| 2023 | AEC-GAN      | GAN       | long TS                       | https://github.com/HBhswl/AEC-GAN                            |
| 2023 | TT-AAE       | GAN       | general TS                    | https://openreview.net/forum?id=fI3y_Dajlca                  |
| 2021 | TimeVAE      | VAE       | general TS                    | https://github.com/abudesai/timeVAE                          |
| 2023 | CRVAE        | VAE       | medical TS & causal discovery | https://github.com/sinhasam/CRVAE                            |
| 2020 | CTFP         | Flow      | general TS                    | https://github.com/BorealisAI/continuous-time-flow-process   |
| 2021 | Fourier Flow | Flow      | general TS                    | https://github.com/ahmedmalaa/Fourier-flows                  |
| 2018 | Neural ODE   | ODE + RNN | general TS                    | https://github.com/rtqichen/torchdiffeq                      |
| 2019 | ODE-RNN      | ODE + RNN | irregular TS                  | https://github.com/YuliaRubanova/latent_ode                  |
| 2021 | Neural SDE   | ODE + GAN | general TS                    | https://github.com/google-research/torchsde                  |
| 2022 | GT-GAN       | ODE + GAN | general (w/ irregular) TS     | https://openreview.net/forum?id=ez6VHWvuXEx                  |
| 2023 | LS4          | ODE + VAE | general (w/ forecasting) TS   | https://github.com/alexzhou907/ls4                           |
| 2016 | WaveNet      | CNN       | speech                        |                                                              |
| 2023 | SGM          | Diffusion | general TS                    | No code available                                            |





## Datasets

TSGBench selects ten real-world datasets from various domains.



| Dataset     | $R$     | $n$    | $l$    | Domain     | Reference                                                    |
| ----------- | ----- | ---- | ---- | ---------- | ------------------------------------------------------------ |
| DLG         | 246   | 20   | 14   | Traffic    | http://archive.ics.uci.edu/dataset/157/dodgers+loop+sensor   |
| Stock       | 3294  | 6    | 24   | Financial  | https://finance.yahoo.com/quote/GOOG/history?p=GOOG          |
| Stock Long  | 3204  | 6    | 125  | Financial  | https://finance.yahoo.com/quote/GOOG/history?p=GOOG          |
| Exchange    | 6715  | 8    | 125  | Financial  | https://github.com/laiguokun/multivariate-time-series-data   |
| Energy      | 17739 | 28   | 24   | Appliances | http://archive.ics.uci.edu/dataset/374/appliances+energy+prediction |
| Energy Long | 17649 | 28   | 125  | Appliances | http://archive.ics.uci.edu/dataset/374/appliances+energy+prediction |
| EEG         | 13366 | 14   | 128  | Medical    | https://archive.ics.uci.edu/dataset/264/eeg+eye+state        |
| HAPT        | 1514  | 6    | 128  | Medical    | https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postura+transitions                                                            |
| Air         | 7731  | 6    | 168  | Sensor     | https://www.microsoft.com/en-us/research/project/urban-air/  |
| Boiler      | 80935 | 11   | 192  | Industrial | https://github.com/DMIRLAB-Group/SASA/tree/main/datasets/Boiler |




## Evaluation Measures

TSGBench considers the following evaluation measures, ranking analysis, and a novel generalization test by Domain Adaptation (DA).

1. Model-based measures
   - Discriminitive Score (DS)
   - Predictive Score (PS)
   - Contextual-FID (C-FID)

2. Feature-based measures
   - Marginal Distribution Difference (MDD)
   - AutoCorrelation Difference (ACD)
   - Skewness Difference (SD)
   - Kurtosis Difference (KD)

3. Distance-based measures
   - Euclidean Distance (ED)
   - Dynamic Time Warping (DTW)

4. Visualization
   - t-SNE
   - Distribution Plot
5. Training Efficiency



## Benchmarking Results


### Main Results

![TSG Benchmarking Results](https://github.com/YihaoAng/TSGBench/blob/main/figures/benchmark_results.png)

### Visualization

![Visualization for TSG Benchmarking by t-SNE and Distribution Plot](https://github.com/YihaoAng/TSGBench/blob/main/figures/visualization_results.png)

### Generalization Test

![Generalization Test](https://github.com/YihaoAng/TSGBench/blob/main/figures/generalization_test_results.png)


## Get Started with TSGBench


### Configuration

Folder `./data` contains the input data, which has the shape of $(R, l, N)$. 

Folder `./data/method_name` contains the generated data, which has the shape of $(R, l, N)$. 

We recommend using `conda` to create a virtual environment.

```
conda create -n tsgbench python=3.7
conda activate tsgbench
conda install --file requirements.txt
```



### Run TSGBench

First, users can first modify the run data preprocessing by 

```bash
python preprocessing/data_preprocess.py
```

Users can also preprocess the data for the generalization test by 

```bash
python preprocessing/data_preprocess_generalization.py
```

Then, users can then generate time series and store the generated time series in `./data/method_name/`.

Finally, users can run evaluations by

```bash
# DS & PS
python ds_ps.py --method_name rgan --dataset_name stock --dataset_state train --gpu_id 0 --gpu_fraction 0.99

# C-FID
python c_fid.py --method_name rgan --dataset_name stock --dataset_state train --gpu_id 0

# MDD, ACD, SD, KD, EU, DTW
python feature_distance_eval.py --method_name rgan --dataset_name stock --dataset_state train --gpu_id 0

# t-SNE, Distribution Plot
python visualization.py --method_name rgan --dataset_name stock --dataset_state train
```
