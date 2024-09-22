import argparse
import yaml
import os
import sys

from src.preprocess import preprocess_data, load_preprocessed_data
from src.generation import generate_data, load_generated_data
from src.evaluation import evaluate_data
from src.utils import write_json_data

project_root = os.path.dirname(os.path.realpath(__file__))

def load_config_from_file(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Pipeline to preprocess, generate, and evaluate data.")

    parser.add_argument("-c","--config", default=f"{project_root}/config/config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("-nlc","--no_load_config", action="store_true", default=False, help="Do not load config from configuration YAML file so you need to specify the config by terminal commands")

    parser.add_argument("-p","--do_preprocessing", action="store_false", default=True, help="Whether to do data preprocessing")
    parser.add_argument("-odp","--original_data_path", default="YOUR_DATA_PATH_HERE", help="Original data path")
    parser.add_argument("-oop","--output_ori_path", default="./data/ori/", help="Path to the output directory")
    parser.add_argument("-dnp","--dataset_name_pre", default="stock", help="Dataset name to store on disc")
    parser.add_argument("-uud","--use_ucr_uea_dataset", action="store_true", default=False, help="Whether to use UCR/UEA dataset")
    parser.add_argument("-udn","--ucr_uea_dataset_name:", default="stock", help="Name of UCR/UEA dataset")
    parser.add_argument("-sl","--seq_length", default=125, type=int, help="The window length of time series")
    parser.add_argument("-vr","--valid_ratio", default=0.1, type=float, help="The ratio of validation set")
    parser.add_argument("-dm","--do_normalization", action="store_false", default=True, help="The window length of time series")
    
    parser.add_argument("-g","--do_generation", action="store_false", default=True, help="Whether to do data generation")
    parser.add_argument("-m","--model", default="TimeVAE", help="Model name to do generation")
    parser.add_argument("-dng","--dataset_name_gen", default="stock", help="Dataset name")
    parser.add_argument("-ogp","--output_gen_path", default="./data/ori/", help="Path to the output directory")
    parser.add_argument("-pp","--pretrain_path", default="./pretrain_model/", help="Path to the output directory")
    parser.add_argument("-ncg","--no_cuda_gen", action="store_false", default=True, help="Disable cuda in generation stage")
    parser.add_argument("-cdg","--cuda_device_gen", default=0, type=int, help="Cuda device index in generation stage")
    parser.add_argument("-ld","--latent_dim", default=64, type=int, help="Latent dimension in TimeVAE model")
    parser.add_argument("-hl","--hidden_layer", default=[32, 64, 128], help="Hidden layer list TimeVAE model")
    
    parser.add_argument("-e","--do_evaluation", action="store_false", default=True, help="Whether to do data evaluation")
    parser.add_argument("-rp","--result_path", default="./result/", help="Path to the result directory")
    parser.add_argument("-nce","--no_cuda_eval", action="store_false", default=True, help="Disable cuda in evaluation stage")
    parser.add_argument("-cde","--cuda_device_eval", default=0, type=int, help="Cuda device index used in evaluation stage")
    parser.add_argument("-ml","--method_list", default='[DS,PS]', help="Evaluation method list")
    parser.add_argument("-id","--iter_disc", default=2000, type=int, help="Iteration in discriminative score")
    parser.add_argument("-ip","--iter_pred", default=5000, type=int, help="Iteration in predictive score")
    parser.add_argument("-rn","--rnn_name", default="lstm", help="RNN name used in DS and PS")
    
    return parser.parse_args()

def main():
    args = parse_command_line_arguments()

    if args.config and not args.no_load_config:
        # Load configuration from YAML file
        config = load_config_from_file(args.config)
    else:
        # Build configuration from command-line arguments
        args_dict = vars(args)
        config = {'preprocessing':{},'generation':{},'evaluation':{}}
        for key, value in args_dict.items():
            if key in ['do_preprocessing','original_data_path','output_ori_path','use_ucr_uea_dataset','ucr_uea_dataset_name','seq_length','valid_ratio','do_normalization']:
                config['preprocessing'][key]=value
            if key == 'dataset_name_pre':
                config['preprocessing']['dataset_name']=value

            if key in ['do_generation','model','output_gen_path','pretrain_path','latent_dim','hidden_layer']:
                config['generation'][key]=value
            if key == 'no_cuda_gen':
                config['generation']['no_cuda']=value
            if key == 'cuda_device_gen':
                config['generation']['cuda_device']=value
            
            if key in ['do_evaluation','result_path','method_list','iter_disc','iter_pred','rnn_name']:
                config['evaluation'][key]=value
            if key == 'no_cuda_eval':
                config['generation']['no_cuda']=value
            if key == 'cuda_device_eval':
                config['generation']['cuda_device']=value
    
    print(config)

    # Execute pipeline steps
    if config['preprocessing'].get('do_preprocessing',True):
        data = preprocess_data(config['preprocessing'])
    else:
        data = load_preprocessed_data(config['preprocessing'])

    # Data preprocessed can be used to train your generation model, assume training process already done

    if config['generation'].get('do_generation',True):
        generated_data = generate_data(config['generation'], data)
    else:
        generated_data = load_generated_data(config['generation'])

    if config['evaluation'].get('do_evaluation',True):
        dataset_name = config['preprocessing'].get('dataset_name','dataset')
        model_name = config['generation'].get('model','TimeVAE')
        if 'dataset_name' not in config['evaluation']:
            config['evaluation']['evaluation'] = dataset_name
        if 'model' not in config['evaluation']:
            config['evaluation']['model'] = model_name
        results = evaluate_data(config['evaluation'],data,generated_data)
        
    print('Program normal end.')

if __name__ == "__main__":
    # Add the project root to sys.path if hasn't
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    main()
