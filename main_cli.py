import argparse
import yaml
import time
import os

from models import ModelManager
from tasks.task_manager import TaskManager, TEMPLATE_PATH

# Set dataset_path or use cmd line args
DATASET_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'data')

def parse_args():
    parser = argparse.ArgumentParser(description='Run in command line')
    # #### Introduce the arguments ####

    # ---- 1. Basic Configuration -----
    parser.add_argument('--task_name', type=str, required=True, 
                        help='str, The result log will be saved in {output_dir}/{task_name}/log/xxxx.yaml')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='str, The result log will be saved in {output_dir}/{task_name}/log/xxxx.yaml')
    parser.add_argument('--train_test', type=str, required=True, 
                        help='str, please type \'train\' or \'test\'. \'train\' mode will train first then test.')
    # [optional]
    parser.add_argument('--debug', default=False, action='store_true', required=False,
                        help='[optional] bool, enable debug mode')
    parser.add_argument('--logger', type=str, default='none', required=False,
                        help='[optional] str, logger, chose one from [\'none\', \'wandb\']. \'none\' logger: just print the log, \'wandb\' logger: use wandb to log the result.')
    parser.add_argument('--device', type=str, default='cuda', required=False,
                        help='[optional] str, device, default=cuda, chose one from [\'cpu\', \'cuda\'].')
    
    # ---- 2. Dataset Configuration -----
    parser.add_argument('--dataset', type=str, required=True,
                        help=f"str, dataset name")
    parser.add_argument('--his_len', type=int, required=True,
                        help='int, his_len, input history length')
    parser.add_argument('--pred_len', type=int, required=True,
                        help='int, pred_len, output prediction length')
    parser.add_argument('--lag_input', type=str, default=[], required=False,
                        help='[optional] list, lag_input=[trend, period, closeness], default=[]')
    parser.add_argument('--data_mode', type=int, default=0, required=True,
                        help='int, data_mode, \nTensorModel: 0:(time, dim1, dim2); 1:(time, dim2, dim1); 2:(time, dim1 x dim2, 1)\nMultiVarModel: 0:(1, time, dim1*dim2, 1); 1:(dim1, time, dim2, 1); 2:(dim2, time, dim1, 1)')
    parser.add_argument('--dataset_base', type=str, default=DATASET_BASE, required=False,
                        help='[optional] str, dataset base path, default=DATASET_BASE')
    # [optional]
    parser.add_argument('--batch_size', type=int, default=256, required=False,
                        help='[optional] int, batch size, default=256, for some dataset, the batch size should small.')
    parser.add_argument('--normalizer', type=str, default='std', required=False,
                        help='[optional] str, normalizer, chose one from [\'none\', \'std\', \'sklearn\'], default=\'std\'')
    
    # ---- 3. Training Configuration -----
    parser.add_argument('--model', type=str, required=True,
                        help='model name')
    # [optional]
    parser.add_argument('--model_path', type=str, default='', required=False,
                        help='[optional] str, model path, for test mode only, load the model from model_path. If mode is train, model_path will be ignored.')
    parser.add_argument('--graph_init', type=str, default='pearson', required=False,
                        help='[optional] str, graph_init, chose one from [\'pearson\', \'inverse_pearson\', \'random\', \'cosine\, \'unit\', default=\'pearson\'')
    parser.add_argument('--seed', type=int, default=2024, required=False,
                        help='[optional] int, random seed, default=2024')
    parser.add_argument('--epochs', type=int, default=256, required=False,
                        help='[optional] int, default=256')
    parser.add_argument('--early_stop_max', type=int, default=16, required=False,
                        help='[optional] int, early_stop_max, default=10')
    parser.add_argument('--early_stop_start_epoch', type=int, default=0, required=False,
                        help='[optional] int, early_stop_start_epoch, ignore early stop in first X epoches, default=0')
    parser.add_argument('--lr_finder', default=False, action='store_true', required=False,
                        help='[optional] bool, enable lr_finder, if lr_finder is enabled, the lr will be set automatically.')
    parser.add_argument('--lr', type=str, default='', required=False,
                        help='[optional] str, learning rate, set \'\' to use default value')
    parser.add_argument('--eps', type=str, default='', required=False,
                        help='[optional] str, set \'\' to use default value')
    parser.add_argument('--weight_decay', type=str, default='', required=False,
                        help='[optional] str, weight decay, set \'\' to use default value')
    parser.add_argument('--scheduler', type=str, default='None', required=False,
                        help="[optional] str, learning rate scheduler, chose one from ['None', 'MultiStepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau']")
    # parse
    args = parser.parse_args()
    return args

def get_config_template(model_name:str):
    model_manager = ModelManager()
    model_type = model_manager.get_model_type(model_name)
    if model_type not in TEMPLATE_PATH:
        raise ValueError(f"model_type {model_type} is not in {TEMPLATE_PATH.keys()}")
    config = yaml.safe_load(open(TEMPLATE_PATH[model_type], 'r'))
    return config

def EnsureDir(output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if __name__=='__main__':
    args = parse_args()

    # make configs from template and args
    model_name = args.model
    basic_config = get_config_template(model_name)

    # update basic_config
    # ---- 1. Basic Configuration -----
    basic_config['project_name'] = args.task_name
    basic_config['output_dir'] = os.path.join(args.output_dir, args.task_name)
    basic_config['mode'] = args.train_test
    basic_config['debug'] = args.debug
    basic_config['logger'] = args.logger
    basic_config['task_device'] = args.device

    # ---- 2. Dataset Configuration -----
    basic_config['dataset_name'] = args.dataset
    basic_config['his_len'] = args.his_len
    basic_config['pred_len'] = args.pred_len
    lag_input = (args.lag_input)
    if lag_input != []:
        lag_input = lag_input.strip('[]').split(',')
        lag_input = [int(i) for i in lag_input]
    basic_config['lag_input'] = lag_input
    basic_config['data_mode'] = args.data_mode
    basic_config['batch_size'] = args.batch_size
    basic_config['normalizer'] = args.normalizer
    basic_config['dataset_base'] = args.dataset_base
    DATASET_BASE = args.dataset_base
    
    # ---- 3. Training Configuration -----
    basic_config['model_name'] = args.model
    basic_config['model_path'] = args.model_path
    basic_config['graph_init'] = args.graph_init
    basic_config['seed'] = args.seed
    basic_config['max_epoch'] = args.epochs
    basic_config['early_stop_max'] = args.early_stop_max
    basic_config['early_stop_start_epoch'] = args.early_stop_start_epoch
    basic_config['lr_finder'] = args.lr_finder
    basic_config['lr'] = args.lr
    basic_config['eps'] = args.eps
    basic_config['weight_decay'] = args.weight_decay
    basic_config['scheduler'] = args.scheduler

    # double check configs
    if basic_config['mode'] == 'train':
        basic_config['model_path'] = ''
    elif basic_config['mode'] == 'test':
        if basic_config['model_path'] == '' and basic_config['model_name'] not in ['HM_TTS', 'HM_MTS', 'HM']:
            raise ValueError("In test mode, model_path should not be empty.")
    if basic_config['lag_input'] != []:
        basic_config['his_len'] = sum(basic_config['lag_input'])
        
    NatureList = ['COVID_DEATHS', 'COVID_CHI', 'COVID_US']
    if basic_config['dataset_name'] in NatureList:
        basic_config['bacth_size'] = 1
    if basic_config['model_name'] in ['HM_TTS', 'HM_MTS', 'HM']:
        basic_config['scheduler'] = 'None'
    # print("=====================================")
    # print("Configs:")
    # for k, v in basic_config.items():
    #     print(f"{k}: {v}")
    # exit()

    # if this task is finished
    if lag_input == []:
        lag_input_str = ""
    else:
        lag_input_str = '-['+"-".join([str(i) for i in lag_input])+']'
    task_path = os.path.join(basic_config['output_dir'], 'checkpoints')
    GraphTTS_prefix = f"{basic_config['dataset_name']}-{basic_config['model_name']}-{basic_config['data_mode']}-{basic_config['his_len']}-{basic_config['pred_len']}{lag_input_str}-{basic_config['graph_init']}-{basic_config['normalizer']}"
    NoGraphTTS_prefix = f"{basic_config['dataset_name']}-{basic_config['model_name']}-{basic_config['data_mode']}-{basic_config['his_len']}-{basic_config['pred_len']}{lag_input_str}-{basic_config['normalizer']}"
    MTS_prefix = f"{basic_config['dataset_name']}-{basic_config['model_name']}-{basic_config['his_len']}-{basic_config['pred_len']}{lag_input_str}-{basic_config['data_mode']}-{basic_config['normalizer']}"
    TTS_dirs = []
    MTS_dirs = []
    if os.path.exists(task_path) and basic_config['mode']=='train':
        for dir_name in os.listdir(task_path):
            if GraphTTS_prefix in dir_name:
                TTS_dirs.append(os.path.join(task_path, dir_name))
            elif NoGraphTTS_prefix in dir_name:
                TTS_dirs.append(os.path.join(task_path, dir_name))
            elif MTS_prefix in dir_name:
                MTS_dirs.append(os.path.join(task_path, dir_name))
        # print(f"Check finished tasks in {task_path}")
        # print(f"Task: {GraphTTS_prefix}")
        # print(f"Task: {NoGraphTTS_prefix}")
        # print(f"Task: {MTS_prefix}")
        for dir_name in TTS_dirs:
            if os.path.exists(os.path.join(dir_name, "model.pth")):
                print(f"This task is finished in {dir_name}, continue to next one...")
                exit()
        for dir_name in MTS_dirs:
            if os.path.exists(os.path.join(dir_name, "run_0", "model.pth")):
                print(f"This task is finished in {dir_name}, continue to next one...")
                exit()
    # timestamp
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    basic_config['timestamp'] = timestamp
    # start
    log_dir = os.path.join(basic_config['output_dir'], 'log')
    EnsureDir(log_dir)

    task_manager = TaskManager('checkpoints', basic_config['output_dir'], dataset_path=basic_config['dataset_base'])
    only_test = True if basic_config['mode']=='test' else False
    res = task_manager.TaskRun(basic_config['dataset_name'], basic_config['model_name'], basic_config, only_test=only_test)

    task_result = {
        'model_name': basic_config['model_name'],
        'dataset_name': basic_config['dataset_name'],
        'data_mode': basic_config['data_mode'],
        'graph_init': basic_config['graph_init'],
        'seed': basic_config['seed'],
        'batch_szie': basic_config['batch_size'],
        'his_len': basic_config['his_len'],
        'pred_len': basic_config['pred_len'],
        'lag_input': basic_config['lag_input'],
        'timestamp': timestamp,
        'result': res,
    }
    save_path = os.path.join(log_dir, f"{basic_config['model_name']}_{basic_config['dataset_name']}_{basic_config['his_len']}-{basic_config['pred_len']}-{lag_input_str}-Mode{basic_config['data_mode']}-{timestamp}.yaml")
    yaml.safe_dump(task_result, open(save_path, 'w'))
