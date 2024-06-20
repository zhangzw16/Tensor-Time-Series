import os
import time
import yaml
import argparse
from models import ModelManager
from tasks.tensor_task import TensorTask

DATASET_PATH = './datasets/Tensor-Time-Series-Dataset/Processed_Data'
TEMPLATE_PATH = './tasks/tensor_tasks_template.yaml'

DatasetMap = {
    "Traffic": ['JONAS_NYC_bike', 'JONAS_NYC_taxi', 'Metr-LA','METRO_HZ', 'METRO_SH','PEMS03', 'PEMS07', 'PEMSBAY'],
    "Natural": ['COVID_DEATHS'],
    "Energy":  ['ETT_hour'],
}

ModelMap = {
    'Tensor':   ['NET3', 'DCRNN', 'GraphWaveNet', 'AGCRN', 'MTGNN', 'TTS_Norm', 'ST_Norm', 'GMRL'],
    'MultiVar': ['TimesNet', 'StemGNN', 'STGCN', 'AutoFormer', 'CrossFormer', 'PatchTST'],
    'Stat':     ['HM'],
}

TensorGraphMap = {
    "prior":   ['NET3', 'DCRNN', 'GraphWaveNet'],
    'learned': ['AGCRN', 'MTGNN'],
    'none':    ['TTS_Norm', 'ST_Norm', 'GMRL'],
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Runner:
    def __init__(self, output_dir:str, config_template:str=TEMPLATE_PATH) -> None:
        self.template = yaml.safe_load(open(config_template, 'r'))
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        ensure_dir(checkpoints_dir)
        self.template['output_dir'] = checkpoints_dir
        log_dir = os.path.join(output_dir, 'log')
        ensure_dir(log_dir)
        self.output_dir = log_dir
        self.results = {}

    def search_pkl(self, dataset_name:str):
        pkl_path = os.path.join(DATASET_PATH, dataset_name)
        files = os.listdir(pkl_path)
        for file in files:
            if file.endswith('.pkl'):
                pkl_path = os.path.join(pkl_path, file)
                break
            raise ValueError(f"no pkl file in {pkl_path}")
        return pkl_path

    # def _run(self, dataset_name, model_type:str, model_list:list,config:dict):
    #     run_config = config.copy()
    #     run_config['dataset_pkl'] = self.search_pkl(dataset_name)
    #     run_config['model_type'] = model_type
    #     run_output = {model_name: "" for model_name in model_list}

    #     for model_name in model_list:
    #         run_config['model_name'] = model_name
    #         try:
    #             task = TensorTask(run_config)
    #             task.train()
    #             result = task.test()
    #             for key in result.keys():
    #                 result[key] = float(result[key])
    #             run_output[model_name] = result
    #         except Exception as exp:
    #             run_output[model_name] = str(exp)
    #     return run_output
    
    def _run(self, dataset_name, model_type:str, model_list:list, config:dict, graph_init:str='pearson'):
        run_config = config.copy()
        run_config['dataset_pkl'] = self.search_pkl(dataset_name)
        run_config['model_type'] = model_type
        run_config['graph_init'] = graph_init
        run_output = {model_name: "" for model_name in model_list}

        for model_name in model_list:
            run_config['model_name'] = model_name
            try:
                task = TensorTask(run_config)
                task.train()
                result = task.test()
                for key in result.keys():
                    result[key] = float(result[key])
                run_output[model_name] = result
            except Exception as exp:
                run_output[model_name] = str(exp)
        return run_output
    
    def auto_run(self, his_len, pred_len, model_type, model_list, dataset_type, data_mode=0):
        run_config = self.template.copy()
        run_config['his_len'] = his_len
        run_config['pred_len'] = pred_len
        run_config['model_type'] = model_type

        if model_type == 'MultiVar':
            run_config['data_mode'] = 2
        else:
            run_config['data_mode'] = data_mode

        dataset_list = DatasetMap[dataset_type]
        time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        auto_run_results = {dataset_name: {} for dataset_name in dataset_list}
        
        # save the config
        auto_run_results['model_type'] = model_type
        auto_run_results['his_len'] = his_len
        auto_run_results['pred_len'] = pred_len
        auto_run_results['data_mode'] = data_mode
        auto_run_results['models'] = model_list

        for dataset_name in dataset_list:
            run_config['dataset_pkl'] = self.search_pkl(dataset_name)
            run_output = self._run(dataset_name, model_type, model_list, run_config)

        # config_output = open(os.path.join(self.output_dir, f'{model_type}_{his_len}_{pred_len}_{time_stamp}.yaml'), 'w')
        # for key in auto_run_results:
        #     yaml.dump(auto_run_results[key], config_output)
        if len(model_list) == 1:
            model_name = model_list[0]
        else:
            model_name = ''
        yaml_path = os.path.join(self.output_dir, f'{model_type}_{model_name}_{dataset_type}_{his_len}_{pred_len}_{time_stamp}.yaml')
        yaml.dump(auto_run_results, open(yaml_path, 'w'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--his_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)

    parser.add_argument('--model_type', type=str, default='Tensor-prior', 
                        help="['Stat', 'MultiVar', 'Tensor-prior', 'Tensor-learned', 'Tensor-none']")
    parser.add_argument('--model_name', type=str, default='',
                        help='specify the model name')
    parser.add_argument('--dataset_type', type=str, default='Traffic',
                        help="['Traffic', 'Natural', 'Energy']")
    parser.add_argument('--data_mode', type=int, default=0, required=False,
                        help='default is 0, and if you chose Multivar model, data_mode will be 2.\n0:(time, dim1, dim2); 1:(time, dim2, dim1); 2:(time, dim1 x dim2, 1)') 
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--config_template', type=str, default=TEMPLATE_PATH)

    args = parser.parse_args()
    
    model_type = args.model_type
    model_name = args.model_name
    if model_name != '':
        model_list = [model_name]
        for key in ModelMap.keys():
            if model_name in ModelMap[key]:
                model_type = key
                break
    else:
        if 'Tensor' in args.model_type:
            splited = model_type.split('-')
            model_type = splited[0]
            graph_type = splited[1]
            model_list = TensorGraphMap[graph_type]
        else:
            model_list = ModelMap[model_type]

    runner = Runner(args.output_dir, args.config_template)
    runner.auto_run(args.his_len, args.pred_len, model_type, model_list, args.dataset_type, args.data_mode)
    



    


        
