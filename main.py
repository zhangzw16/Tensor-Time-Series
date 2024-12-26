import os
import yaml
import time

from models import ModelManager
from tasks.task_manager import TaskManager, TEMPLATE_PATH

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
# "NET3" "DCRNN" "AGCRN" "STC_GNN" "GraphWaveNet"
# "MTGNN" "ST_Norm" "TTS_Norm" "GMRL" "GCGRU" "Mamba"
# "TimesNet" "StemGNN" "AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" "STID"
if __name__=='__main__':
    # set model and dataset
    model_name = 'PatchTST'
    dataset_name = 'METRO_HZ'
    basic_config = get_config_template(model_name)
    # update basic_config
    # DATASET_BASE = 
    DATASET_BASE = '/nas/datasets/Tensor-Time-Series-Dataset/Processed_Data'
    # ---- 1. Basic Configuration -----
    basic_config['project_name'] = 'main_debug'
    output_dir = '/data4t/zjx_dataset/workspace/Tensor-Time-Series/nas_logs'
    basic_config['output_dir'] = os.path.join(output_dir, basic_config['project_name'])
    basic_config['mode'] = 'train'
    basic_config['debug'] = True
    basic_config['logger'] = 'none'
    basic_config['task_device'] = 'cuda'

    # ---- 2. Dataset Configuration -----
    basic_config['dataset_name'] = dataset_name
    basic_config['his_len'] = 6
    basic_config['pred_len'] = 12
    basic_config['lag_input'] = []
    basic_config['data_mode'] = 0
    basic_config['batch_size'] = 2
    basic_config['normalizer'] = 'sklearn'
    
    # ---- 3. Training Configuration -----
    basic_config['model_name'] = model_name
    basic_config['model_path'] = ''
    basic_config['graph_init'] = 'unit'
    basic_config['seed'] = 2024
    basic_config['max_epoch'] = 2024
    basic_config['early_stop_max'] = 32
    basic_config['early_stop_start_epoch'] = 0
    basic_config['lr_finder'] = True 
    # basic_config['lr'] = '1e-7'
    basic_config['lr'] = '1e-4'
    basic_config['eps'] = '1e-8'
    basic_config['weight_decay'] = '1e-3'
    basic_config['scheduler'] = 'ReduceLROnPlateau'
    basic_config['model_path'] = '/home/ysc/workspace/output/main/checkpoints/weather-PatchTST-512-96-0-std-2024-11-14-15:09:45/run_0/checkpoint_30.pth'

    # double check
    if basic_config['mode'] == 'train':
        basic_config['model_path'] = ''
    elif basic_config['mode'] == 'test':
        if basic_config['model_path'] == '':
            raise ValueError("In test mode, model_path should not be empty.")
    if basic_config['lag_input'] != []:
        basic_config['his_len'] = sum(basic_config['lag_input'])
    lag_input = basic_config['lag_input']
    if lag_input == []:
        lag_input_str = ""
    else:
        lag_input_str = '-['+"-".join([str(i) for i in lag_input])+']'

    # timestamp
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    basic_config['timestamp'] = timestamp
    # start
    log_dir = os.path.join(basic_config['output_dir'], 'log')
    EnsureDir(log_dir)

    task_manager = TaskManager('checkpoints', basic_config['output_dir'], dataset_path=DATASET_BASE)
    only_test = True if basic_config['mode']=='test' else False
    res = task_manager.TaskRun(basic_config['dataset_name'], basic_config['model_name'], basic_config, only_test=only_test)

    task_result = {
        'model_name': basic_config['model_name'],
        'dataset_name': basic_config['dataset_name'],
        'data_mode': basic_config['data_mode'],
        'graph_init': basic_config['graph_init'],
        'seed': basic_config['seed'],
        'his_len': basic_config['his_len'],
        'pred_len': basic_config['pred_len'],
        'lag_input': basic_config['lag_input'],
        'timestamp': timestamp,
        'result': res,
    }
    save_path = os.path.join(log_dir, f"{basic_config['model_name']}_{basic_config['dataset_name']}_{basic_config['his_len']}-{basic_config['pred_len']}{lag_input_str}-Mode{basic_config['data_mode']}-{timestamp}.yaml")
    yaml.safe_dump(task_result, open(save_path, 'w'))
