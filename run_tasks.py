import argparse
import yaml
import time
import os

from tasks.task_manager import TaskManager, TEMPLATE_PATH

# Set dataset_path !!!
DATASET_BASE = '/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/datasets/data'
# Dataset List
DatasetMap = {
    "Traffic": ['JONAS_NYC_bike', 'JONAS_NYC_taxi', 'Metr-LA','METRO_HZ', 'METRO_SH','PEMS03', 'PEMS07'],
    "Natural": ['COVID_DEATHS'],
    "Energy":  ['ETT_hour', 'electricity'],
    "Weather": ['weather', 'Jena_climate'],
    "Finance": ['nasdaq100', 'electricity'],
}
# Model List
MTS_ModelList = ['TimesNet', 'StemGNN', 'AutoFormer', 'CrossFormer', 'PatchTST']
TTS_ModelList = ['NET3', 'DCRNN', 'GraphWaveNet', 'AGCRN', 'MTGNN', 'TTS_Norm', 'ST_Norm', 'GMRL']
GraphModelMap = {
    "prior":   ['NET3', 'DCRNN', 'GraphWaveNet'],
    'learned': ['AGCRN', 'MTGNN'],
    'none':    ['TTS_Norm', 'ST_Norm'],
}

'''
Tools
'''
def config_loader(model_type):
    if model_type not in TEMPLATE_PATH:
        raise ValueError(f"Model type {model_type} is not supported.")
    config = yaml.safe_load(open(TEMPLATE_PATH[model_type], 'r'))
    return config
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
TASK_REGISTRY = {}
def register_task(name=None):
    def decorator(func):
        task_name = name if name else func.__name__
        TASK_REGISTRY[task_name] = func
        return func
    return decorator
'''
Task1: MTS_Task
Pram:
    - his_len
    - pred_len
    - data_mode: (time, dim1, dim2) --> (time_series_num, time_range, dim, 1)
        + 0: (1, time, dim1*dim2, 1)
        + 1: (dim1, time, dim2, 1)
        + 2: (dim2, time, dim1, 1)
    - batch_size
    - project_name
    - dataset_list
    - output_dir
Output:
    - log file saved in 'output_dir/project_name/log/xxxx.yaml'
    - checkpoints saved in 'output_dir/project_name/checkpoints/task_id/model.pth'
    - model_config snapshot saved in 'output_dir/project_name/checkpoints/task_id/configs.yml'
'''
@register_task('MTS_Task')
def MTS_TasksRun(his_len:int, pred_len:int, data_mode:int, batch_size:int, 
                 project_name:str, dataset_list:list, output_dir:str, only_test:bool):
    base_dir = os.path.join(output_dir, project_name)
    # logger configuration
    log_dir = os.path.join(base_dir, 'log')
    ensure_dir(log_dir)
    # task configuration
    # model_output_dir = os.path.join(output_dir, 'checkpoints')
    # ensure_dir(model_output_dir)
    task_config = config_loader('MultiVar')
    task_config['output_dir'] = base_dir
    task_config['his_len'] = his_len
    task_config['pred_len'] = pred_len
    task_config['batch_size'] = batch_size
    task_config['data_mode'] = data_mode
    # start to run
    task_results = {dataset_name: {} for dataset_name in dataset_list}
    manager = TaskManager('checkpoints', base_dir)
    for model_name in MTS_ModelList:
        for dataset_name in dataset_list:
            res = manager.TaskRun(dataset_name, model_name, task_config, only_test=only_test)
            task_results[dataset_name][model_name] = res
    # dump results
    timestamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    task_results['timestamp'] = timestamp
    task_results['model_list'] = MTS_ModelList
    task_results['dataset_list'] = dataset_list
    task_results['model_type'] = 'MultiVar'
    task_results['his_len'] = his_len
    task_results['pred_len'] = pred_len
    task_results['data_mode'] = data_mode
    yaml_path = os.path.join(log_dir, f'MultiVar-{his_len}-{pred_len}-{data_mode}-{timestamp}.yaml')
    yaml.dump(task_results, open(yaml_path, 'w'))

'''
Task2: TTS_Task
Param:
    - his_len
    - pred_len
    - data_mode: (time, dim1, dim2) 
        + 0: (time, dim1, dim2)
        + 1: (time, dim2, dim1)
        + 2: (time, dim1*dim2, 1)
    - batch_size
    - project_name
    - dataset_list
    - output_dir
    - graph_init is 'pearson' in this task.
Output:
    - log file saved in 'output_dir/project_name/log/xxxx.yaml'
    - checkpoints saved in 'output_dir/project_name/checkpoints/task_id/model.pth'
    - model_config snapshot saved in 'output_dir/project_name/checkpoints/task_id/configs.yml'
'''
@register_task('TTS_Task')
def TTS_TasksRun(his_len:int, pred_len:int, data_mode:int, batch_size:int, 
                 project_name:str, dataset_list:list, output_dir:str, only_test:bool):
    base_dir = os.path.join(output_dir, project_name)
    # logger configuration
    log_dir = os.path.join(base_dir, 'log')
    ensure_dir(log_dir)
    # task configuration
    # model_output_dir = os.path.join(output_dir, 'checkpoints')
    # ensure_dir(model_output_dir)
    task_config = config_loader('Tensor')
    task_config['output_dir'] = base_dir
    task_config['his_len'] = his_len
    task_config['pred_len'] = pred_len
    task_config['batch_size'] = batch_size
    task_config['data_mode'] = data_mode
    task_config['graph_init'] = 'pearson'
    # start to run
    task_results = {dataset_name: {} for dataset_name in dataset_list}
    manager = TaskManager('checkpoints', base_dir)
    for model_name in TTS_ModelList:
        for dataset_name in dataset_list:
            res = manager.TaskRun(dataset_name, model_name, task_config, only_test=only_test)
            task_results[dataset_name][model_name] = res
    # dump results
    timestamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    task_results['timestamp'] = timestamp
    task_results['model_list'] = TTS_ModelList
    task_results['dataset_list'] = dataset_list
    task_results['model_type'] = 'Tensor'
    task_results['his_len'] = his_len
    task_results['pred_len'] = pred_len
    task_results['data_mode'] = data_mode
    task_results['graph_init'] = 'pearson'
    yaml_path = os.path.join(log_dir, f'Tensor-{his_len}-{pred_len}-{data_mode}-{timestamp}.yaml')
    yaml.dump(task_results, open(yaml_path, 'w'))
'''
Task3: GraphTask-Prior
    - only for prior graph tensor model
    - to test which graph initialization is better
Param:
    - his_len
    - pred_len
    - data_mode: the same as 'TTS_Task'
    - batch_size
    - project_name
    - dataset_list
    - output_dir
    - graph_init: ['inverse_pearson', 'random']
Output:
    - log file saved in 'output_dir/project_name/log/xxxx.yaml'
    - checkpoints saved in 'output_dir/project_name/checkpoints/task_id/model.pth'
    - model_config snapshot saved in 'output_dir/project_name/checkpoints/task_id/configs.yml'
'''
@register_task('Graph_Init_Task')
def Graph_Prior_TasksRun(his_len:int, pred_len:int, data_mode:int, batch_size:int, 
                   project_name:str, dataset_list:list, output_dir:str, only_test:bool, graph_init:str):
    base_dir = os.path.join(output_dir, project_name)
    # logger configuration
    log_dir = os.path.join(base_dir, 'log')
    ensure_dir(log_dir)
    # task configuration
    # model_output_dir = os.path.join(output_dir, 'checkpoints')
    # ensure_dir(model_output_dir)
    task_config = config_loader('Tensor')
    task_config['output_dir'] = base_dir
    task_config['his_len'] = his_len
    task_config['pred_len'] = pred_len
    task_config['batch_size'] = batch_size
    task_config['data_mode'] = data_mode
    task_config['graph_init'] = graph_init
    # start to run
    task_results = {dataset_name: {} for dataset_name in dataset_list}
    manager = TaskManager('checkpoints', base_dir)
    for model_name in GraphModelMap['prior']:
        for dataset_name in dataset_list:
            res = manager.TaskRun(dataset_name, model_name, task_config, only_test=only_test)
            task_results[dataset_name][model_name] = res
    # dump results
    timestamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    task_results['timestamp'] = timestamp
    task_results['model_list'] = GraphModelMap[graph_init]
    task_results['dataset_list'] = dataset_list
    task_results['model_type'] = 'Tensor'
    task_results['his_len'] = his_len
    task_results['pred_len'] = pred_len
    task_results['data_mode'] = data_mode
    task_results['graph_init'] = graph_init
    yaml_path = os.path.join(log_dir, f'Tensor-Graph-{graph_init}-{his_len}-{pred_len}-{data_mode}-{timestamp}.yaml')
    yaml.dump(task_results, open(yaml_path, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # args
    parser.add_argument('--his_len', type=int, required=True,
                        help='input history length')
    parser.add_argument('--pred_len', type=int, required=True,
                        help='output prediction length')
    parser.add_argument('--data_mode', type=int, default=0,
                        help='TensorModel: 0:(time, dim1, dim2); 1:(time, dim2, dim1); 2:(time, dim1 x dim2, 1)\nMultiVarModel: 0:(1, time, dim1*dim2, 1); 1:(dim1, time, dim2, 1); 2:(dim2, time, dim1, 1)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size, default=8')
    parser.add_argument('--dataset', type=str, required=True,
                        help=f"key in DatasetMap: {DatasetMap.keys()}")
    parser.add_argument('--task_name', type=str, required=True,
                        help=f'task name chosen from {TASK_REGISTRY.keys()}. \nThe log file will be saved in output_dir/task_name/log/xxxx.yaml')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output path. The log file will be saved in output_dir/project_name/log/xxxx.yaml')
    parser.add_argument('--graph_init', type=str, default='pearson', required=False,
                        help='[optional] only for Graph_Init_Task, graph initialization method: [pearson, inverse_pearson, random]')
    parser.add_argument('--only_test', type=str, default='False', required=False,
                        help='[optional] test only')
    # parse
    args = parser.parse_args()
    # check if the value is valid
    if args.task_name not in TASK_REGISTRY:
        raise ValueError(f"Task {args.task_name} is not supported.")
    if args.dataset not in DatasetMap:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    
    task_func = TASK_REGISTRY[args.task_name]
    task_param = {}
    task_param['his_len'] = args.his_len
    task_param['pred_len'] = args.pred_len
    task_param['data_mode'] = args.data_mode
    task_param['batch_size'] = args.batch_size
    task_param['project_name'] = args.task_name
    task_param['dataset_list'] = DatasetMap[args.dataset]
    task_param['output_dir'] = args.output_dir
    task_param['only_test'] = False if args.only_test == 'False' else True
    if args.task_name == 'Graph_Init_Task':
        task_param['graph_init'] = args.graph_init
    # start to run
    # print(task_param)
    task_func(**task_param)


