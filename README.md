<div align= "center">
  <h1> TSGBench: Time Series Generation Benchmark </h1>
</div>



**[TSGBench](https://www.vldb.org/pvldb/vol17/p305-huang.pdf)** is the inaugural TSG benchmark designed for the Time Series Generation (TSG) task. We are excited to share that TSGBench has received the **Best Research Paper Award Nomination** at VLDB 2024 &#x1F3C6;


**[TSGAssist](https://www.vldb.org/pvldb/vol17/p4309-huang.pdf)** is an interactive assistant that integrates the strengths of TSGBench and utilizes Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) for TSG recommendations and benchmarking &#x1F916;&#x1F4CA;


**We are actively exploring industrial collaborations in time series analytics. Please feel free to reach out (yihao_ang AT comp.nus.edu.sg) if interested** &#x1F91D;&#x2728;



<p align="center">
<img src="https://github.com/YihaoAng/TSGBench/blob/main/figures/ranking.png" alt="drawing" width="600"/>
</p>



## Table of Contents

- [Overview of TSGBench](#overview-of-tsgbench)
    - [Time Series Generation (TSG)](#time-series-generation-tsg)
    - [TSG Methods](#tsg-methods)
    - [TSG Datasets](#tsg-datasets)
    - [TSG Evaluation Measures](#tsg-evaluation-measures)
    - [Benchmarking Results](#benchmarking-results)
      - [Main Results](#main-results)
      - [Visualization](#visualization)
      - [Generalization Test](#generalization-test)
- [Overview of TSGAssist](#overview-of-tsgassist)
- [Getting Started with TSGBench](#getting-started-with-tsgbench)
  - [Configuration](#configuration)
  - [Running TSGBench](#running-tsgbench)




## Overview of TSGBench

![Overall Architecture of TSGBench](https://github.com/YihaoAng/TSGBench/blob/main/figures/overall_architecture.png)


### Time Series Generation (TSG)

**Time Series Generation (TSG)** is crucial in a range of applications, including data augmentation, anomaly detection, and privacy preservation. Given an input time series, TSG aims to produce time series akin to the original, preserving temporal dependencies and dimensional correlations while ensuring the generated time series remains useful for various downstream tasks.



### TSG Methods

TSGBench surveys a diverse range of Time Series Generation (TSG) methods by different backbone models and their specialties. The table below provides an overview of these methods along with their references.


| Time | Paper            | Model     | Specialty                     | Source Codes                                                    |
| ---- | ---------------- | --------- | ----------------------------- | ------------------------------------------------------------ |
| 2016 | C-RNN-GAN        | GAN       | music                         | https://github.com/olofmogren/c-rnn-gan                      |
| 2017 | RGAN             | GAN       | general (w/ medical) TS       | https://github.com/ratschlab/RGAN                            |
| 2018 | T-CGAN           | GAN       | irregular TS                  | https://github.com/gioramponi/GAN_Time_Series                |
| 2019 | WaveGAN          | GAN       | audio                         | https://github.com/chrisdonahue/wavegan                      |
| 2019 | TimeGAN          | GAN       | general TS                    | https://github.com/jsyoon0823/TimeGAN                        |
| 2020 | TSGAN            | GAN       | general TS                    | Community implementation: https://github.com/Yashkataria/CGAN-for-time-series |
| 2020 | DoppelGANger     | GAN       | general TS                    | https://github.com/fjxmlzn/DoppelGANger                      |
| 2020 | SigCWGAN         | GAN       | long financial TS             | https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs |
| 2020 | Quant GANs       | GAN       | long financial TS             | Community implementations: https://github.com/ICascha/QuantGANs-replication, https://github.com/JamesSullivan/temporalCN |
| 2020 | COT-GAN          | GAN       | TS and video                  | https://github.com/tianlinxu312/cot-gan                      |
| 2021 | Sig-WGAN         | GAN       | financial TS                  | https://github.com/SigCGANs/Sig-Wasserstein-GANs             |
| 2021 | TimeGCI          | GAN       | general TS                    | No code available                                            |
| 2021 | RTSGAN           | GAN       | general (w/ incomplete) TS    | https://github.com/acphile/RTSGAN                            |
| 2022 | PSA-GAN          | GAN       | general (w/ forecasting) TS   | https://github.com/mbohlkeschneider/psa-gan                  |
| 2022 | CEGEN            | GAN       | general TS                    | No code available                                            |
| 2022 | TTS-GAN          | GAN       | general TS                    | https://github.com/imics-lab/tts-gan                         |
| 2022 | TsT-GAN          | GAN       | general TS                    | No code available                                            |
| 2022 | COSCI-GAN        | GAN       | general TS                    | https://github.com/aliseyfi75/COSCI-GAN                      |
| 2023 | AEC-GAN          | GAN       | long TS                       | https://github.com/HBhswl/AEC-GAN                            |
| 2023 | TT-AAE           | GAN       | general TS                    | https://openreview.net/forum?id=fI3y_Dajlca                  |
| 2021 | TimeVAE          | VAE       | general TS                    | https://github.com/abudesai/timeVAE                          |
| 2023 | CRVAE            | VAE       | medical TS & causal discovery | https://github.com/sinhasam/CRVAE                            |
| 2023 | TimeVQVAE        | VAE       | general TS                    | https://github.com/ML4ITS/TimeVQVAE                          |
| 2023 | TimeVQVAE w/ ESS | VAE       | general TS                    | https://github.com/ML4ITS/TimeVQVAE?tab=readme-ov-file#enhanced-sampling-scheme-2 |
| 2023 | KVAE             | VAE       | general (w/ irregular) TS     | No code available                                            |
| 2020 | CTFP             | Flow      | general TS                    | https://github.com/BorealisAI/continuous-time-flow-process   |
| 2021 | Fourier Flow     | Flow      | general TS                    | https://github.com/ahmedmalaa/Fourier-flows                  |
| 2018 | Neural ODE       | ODE + RNN | general TS                    | https://github.com/rtqichen/torchdiffeq                      |
| 2019 | ODE-RNN          | ODE + RNN | irregular TS                  | https://github.com/YuliaRubanova/latent_ode                  |
| 2021 | Neural SDE       | ODE + GAN | general TS                    | https://github.com/google-research/torchsde                  |
| 2022 | GT-GAN           | ODE + GAN | general (w/ irregular) TS     | https://openreview.net/forum?id=ez6VHWvuXEx                  |
| 2023 | LS4              | ODE + VAE | general (w/ forecasting) TS   | https://github.com/alexzhou907/ls4                           |
| 2023 | SGM              | Diffusion | general TS                    | No code available                                            |





### TSG Datasets

TSGBench selects ten real-world datasets from various domains, ensuring a wide coverage of scenarios for TSG evaluation. Here, $R$ is the number of sub-matrics after preprocessing, $l$ is the series length, and $N$ is the number of series in the sub-matrics.

| Dataset     | $R$   | $l$  | $N$  | Domain     | Link                                                       |
| ----------- | ----- | ---- | ---- | ---------- | ------------------------------------------------------------------- |
| DLG         | 246   | 14   | 20   | Traffic    | http://archive.ics.uci.edu/dataset/157/dodgers+loop+sensor          |
| Stock       | 3294  | 24   | 6    | Financial  | https://finance.yahoo.com/quote/GOOG/history?p=GOOG                 |
| Stock Long  | 3204  | 125  | 6    | Financial  | https://finance.yahoo.com/quote/GOOG/history?p=GOOG                 |
| Exchange    | 6715  | 125  | 8    | Financial  | https://github.com/laiguokun/multivariate-time-series-data          |
| Energy      | 17739 | 24   | 28   | Appliances | http://archive.ics.uci.edu/dataset/374/appliances+energy+prediction |
| Energy Long | 17649 | 125  | 28   | Appliances | http://archive.ics.uci.edu/dataset/374/appliances+energy+prediction |
| EEG         | 13366 | 128  | 14   | Medical    | https://archive.ics.uci.edu/dataset/264/eeg+eye+state               |
| HAPT        | 1514  | 128  | 6    | Medical    | https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postura+transitions |
| Air         | 7731  | 168  | 6    | Sensor     | https://www.microsoft.com/en-us/research/project/urban-air/         |
| Boiler      | 80935 | 192  | 11   | Industrial | https://github.com/DMIRLAB-Group/SASA/tree/main/datasets/Boiler     |




### TSG Evaluation Measures

TSGBench considers the following evaluation measures, ranking analysis, and a novel generalization test by Domain Adaptation (DA).

1. Model-based Measures
   - Discriminitive Score (DS)
   - Predictive Score (PS)
   - Contextual-FID (C-FID)
2. Feature-based Measures
   - Marginal Distribution Difference (MDD)
   - AutoCorrelation Difference (ACD)
   - Skewness Difference (SD)
   - Kurtosis Difference (KD)
3. Distance-based Measures
   - Euclidean Distance (ED)
   - Dynamic Time Warping (DTW)
4. Visualization
   - t-SNE
   - Distribution Plot
5. Training Efficiency
   - Training Time



### Benchmarking Results


#### Main Results

![TSG Benchmarking Results](https://github.com/YihaoAng/TSGBench/blob/main/figures/benchmark_results.png)

#### Visualization

![Visualization for TSG Benchmarking by t-SNE and Distribution Plot](https://github.com/YihaoAng/TSGBench/blob/main/figures/visualization_results.png)

#### Generalization Test

![Generalization Test](https://github.com/YihaoAng/TSGBench/blob/main/figures/generalization_test_results.png)



## Overview of TSGAssist

TSGAssist is an interactive assistant harnessing LLMs and RAG for time series generation recommendations and benchmarking.
- It offers multi-round personalized recommendations through a conversational interface that bridges the cognitive gap,
- It enables the direct application and instant evaluation of users' data, providing practical insights into the effectiveness of various methods.




<p align="center">
  <img src="https://github.com/YihaoAng/TSGBench/blob/main/figures/tsgassist_sc1.png" alt="Screenshot of TSGAssist 1" width="45%" />
  <img src="https://github.com/YihaoAng/TSGBench/blob/main/figures/tsgassist_sc2.png" alt="Screenshot of TSGAssist 2" width="45%" />
</p>




## Getting Started with TSGBench


### Configuration

We recommend using `conda` to create a virtual environment for TSGBench.

```
conda create -n tsgbench python=3.7
conda activate tsgbench
conda install --file requirements.txt
```
The configuration file `./config/config.yaml` contains various settings to run TSGBench. It is structured into the following sections:

* **Preprocessing**: Configures data preprocessing. Specify the input data path using the `preprocessing.original_data_path` and the output path for processed data using `preprocessing.output_ori_path`.
* **Generation**: Contains the settings related to data generation.
* **Evaluation**: Includes the parameters required for evaluating the model's performance.




### Running TSGBench

1. **Set Input Data**: Update the `preprocessing.original_data_path` in `config.yaml` to specify the location of your input data.

2. **Run TSGBench**: Execute the main script by running `python ./main.py`. By default, this will run the preprocessing, generation, and evaluation stages in sequence. You can skip or adjust these steps by modifying the relevant sections in the configuration file. In particular,

    (1) **Preprocessing**: During preprocessing, data is processed and saved to the path specified by `preprocessing.output_ori_path` in the configuration file.

    (2) **Generation**: Place your designated model structure under the `./model` directory. In `./src/generation`, point the model entry to your model. If necessary, provide pretrained parameters by specifying them under `generation.pretrain_path`. Generated data will be saved at `generation.output_gen_path`.


    (3) **Evaluation**: Select specific evaluation measures by updating the `evaluation.method_list` in the configuration file. The evaluation results will be saved to the path specified in `evaluation.result_path`.


## References

Please consider citing our work if you use TSGBench (and/or TSGAssist) in your research:

```bibtex
# TSGBench
@article{ang2023tsgbench,
  title        = {TSGBench: Time Series Generation Benchmark},
  author       = {Ang, Yihao and Huang, Qiang and Bao, Yifan and Tung, Anthony KH and Huang, Zhiyong},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {17},
  number       = {3},
  pages        = {305--318},
  year         = {2023}
}

# TSGAssist
@article{ang2024tsgassist,
  title        = {TSGAssist: An Interactive Assistant Harnessing LLMs and RAG for Time Series Generation Recommendations and Benchmarking},
  author       = {Ang, Yihao and Bao, Yifan and Huang, Qiang and Tung, Anthony KH and Huang, Zhiyong},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {17},
  number       = {12},
  pages        = {4309--4312},
  year         = {2024}
}
```
