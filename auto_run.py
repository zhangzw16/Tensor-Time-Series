import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time
import yaml
from models import ModelManager
from tasks.tensor_task import TensorTask

DATASET_PATH = './datasets/Processed_Data'
TEMPLATE_PATH = './tasks/tensor_tasks_template.yaml'
data_mode = 'Random_Init_2'


DatasetMap = {
    "Traffic": ['JONAS_NYC_bike', 'JONAS_NYC_taxi', 'Metr-LA','METRO_HZ', 'METRO_SH','PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMS20', 'PEMSBAY'],
    "Natrual": ['COVID_DEATHS'],
    "Energy":  ['ETT_hour','electricity'],
    "Finance": ['crypto12', 'nasdaq100'],
    "Weather": ['weather', 'Jena_climate'],
}

ModelMap = {
    'Tensor':   ['NET3', 'DCRNN', 'GraphWaveNet', 'AGCRN', 'MTGNN', 'TTS_Norm', 'ST_Norm', 'GMRL'],
    # 'MultiVar': ['TimesNet', 'StemGNN', 'STGCN', 'AutoFormer', 'CrossFormer', 'PatchTST'],
    'MultiVar': ['PatchTST'],
    'Stat':     ['HM'],
    # 'NET3':     ['NET3_MLP'],
    'PatchTST': ['PatchTST'],
    'NET3':       ['NET3'],
    
}

TensorGraphMap = {
    "prior":   ['NET3', 'DCRNN', 'GraphWaveNet'],
    'learned': ['AGCRN', 'MTGNN'],
    'none':    ['TTS_Norm', 'ST_Norm', 'GMRL'],
    # 'NET3':     ['NET3_MLP'],
    # 'PatchTST': ['PatchTST'],
    'NET3':       ['NET3'],
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class AutoRunner:
    def __init__(self, output_dir:str, config_template:str=TEMPLATE_PATH) -> None:
        self.template = yaml.safe_load(open(config_template, 'r'))
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        self.template['output_dir'] = checkpoints_dir
        ensure_dir(checkpoints_dir)
        log_dir = os.path.join(output_dir, 'log')
        self.output_dir = log_dir
        ensure_dir(log_dir)
        self.results = {}
        # print(checkpoints_dir, log_dir);exit()
    
    def _run(self, dataset_name, model_type:str, model_list:list,config:dict):
        run_config = config.copy()
        run_config['dataset_pkl'] = self.search_pkl(dataset_name)
        run_config['model_type'] = model_type
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

    def search_pkl(self, dataset_name:str):
        pkl_path = os.path.join(DATASET_PATH, dataset_name)
        files = os.listdir(pkl_path)
        for file in files:
            if file.endswith('.pkl'):
                pkl_path = os.path.join(pkl_path, file)
                break
            raise ValueError(f"no pkl file in {pkl_path}")
        return pkl_path
    
    def run_graph(self, his_len, pre_len, model_type:str, model_list:list, dataset_list:list, 
                  graph_init:str='random', run_task_name:str='Tensor-Graph'):
        graph_config = self.template.copy()
        graph_config['his_len'] = his_len
        graph_config['pred_len'] = pre_len
        graph_config['graph_init'] = graph_init
        graph_config['model_type'] = model_type
        data_mode = 2
        graph_config['data_mode'] = data_mode
        time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
        # output file
        # output_file = open(os.path.join(self.output_dir, f'{run_task_name}_{graph_init}_{time_stamp}.txt'), 'w')
        graph_results = {dataset_name: {} for dataset_name in dataset_list}

        for dataset_name in dataset_list:
            graph_config['project_name'] = f'{run_task_name}_{dataset_name}_{his_len}_{pre_len}_{graph_init}'
            res = self._run(dataset_name, model_type, model_list, graph_config)
            graph_results[dataset_name] = res

        graph_results['models'] = model_list
        graph_results['model_type'] = model_type
        graph_results['datasets'] = dataset_list
        graph_results['his_pred'] = [his_len, pre_len]
        graph_results['data_mode'] = data_mode
        yaml.dump(graph_results, open(os.path.join(self.output_dir, f'{run_task_name}_{graph_init}_{time_stamp}.yaml'), 'w'))
    
    def auto_run(self, his_len, pre_len, model_type:str, model_list:list, dataset_list:list, run_task_name:str='Auto-Run'):
        run_config = self.template.copy()
        run_config['his_len'] = his_len
        run_config['pred_len'] = pre_len
        run_config['model_type'] = model_type
        # [time, dim1*dim2, 1]
        data_mode = 2
        run_config['data_mode'] = data_mode
        time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
        auto_run_results = {dataset_name: {} for dataset_name in dataset_list}

        for dataset_name in dataset_list:
            run_config['project_name'] = f'{run_task_name}_{dataset_name}_{his_len}_{pre_len}'
            auto_run_results[dataset_name] = self._run(dataset_name, model_type, model_list, run_config)

        auto_run_results['models'] = model_list
        auto_run_results['model_type'] = model_type
        auto_run_results['datasets'] = dataset_list
        auto_run_results['his_pred'] = [his_len, pre_len]       
        auto_run_results['data_mode'] = data_mode
        yaml.dump(auto_run_results, open(os.path.join(self.output_dir, f'{run_task_name}_{time_stamp}.yaml'), 'w'))
        # print(auto_run_results)
def run_by_list(his_len, pred_len, out_dir, model_type, model_list):
    auto_runner = AutoRunner(out_dir)
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list)
    auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_Traffic_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_Natrual_{his_len}_{pred_len}')
    auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_Energy_{his_len}_{pred_len}')
    auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Finance'], run_task_name=f'{model_type}_Finance_{his_len}_{pred_len}')
    auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Weather'], run_task_name=f'{model_type}_Weather_{his_len}_{pred_len}')

def run_all(his_len, pred_len, out_dir, task=0):
    auto_runner = AutoRunner(out_dir)
    if task == 0:
        model_type = 'Stat'
        model_list = ModelMap[model_type]
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_Traffic_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_Natrual_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_Energy_{his_len}_{pred_len}')
    elif task == 1:
        model_type = 'MultiVar'
        model_list = ModelMap[model_type]
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_Traffic_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_Natrual_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_Energy_{his_len}_{pred_len}')
    elif task == 2:
        model_type = 'Tensor'
        model_list = TensorGraphMap['none']
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_ng_Traffic_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_ng_Natrual_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_ng_Energy_{his_len}_{pred_len}')
    elif task == 3:
        model_type = 'Tensor'
        model_list = TensorGraphMap['learned']
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_learned_Traffic_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_learned_Natrual_{his_len}_{pred_len}')
        auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_learned_Energy_{his_len}_{pred_len}') 

def run_prior_graph(his_len, pred_len, out_dir, graph_init):
    auto_runner = AutoRunner(out_dir)
    model_type = 'Tensor'
    # auto_runner.run_graph(his_len, pred_len, model_type, TensorGraphMap['prior'], DatasetMap['Traffic'], graph_init, run_task_name=f'{model_type}_{graph_init}_prior_Traffic_{his_len}_{pred_len}')
    # auto_runner.run_graph(his_len, pred_len, model_type, TensorGraphMap['prior'], DatasetMap['Natrual'], graph_init, run_task_name=f'{model_type}_{graph_init}_Traffic_{his_len}_{pred_len}') 
    # auto_runner.run_graph(his_len, pred_len, model_type, TensorGraphMap['prior'], DatasetMap['Energy'], graph_init, run_task_name=f'{model_type}_{graph_init}_Traffic_{his_len}_{pred_len}')
    model_list = TensorGraphMap['prior']
    idx=0
    auto_runner.run_graph(his_len, pred_len, model_type, [model_list[idx]], DatasetMap['Traffic'], graph_init, run_task_name=f'{model_type}_prior_Traffic_{model_list[idx]}_{graph_init}_{his_len}_{pred_len}')
    auto_runner.run_graph(his_len, pred_len, model_type, [model_list[idx]], DatasetMap['Natrual'], graph_init, run_task_name=f'{model_type}_prior_Natrual_{model_list[idx]}_{graph_init}_{his_len}_{pred_len}') 
    auto_runner.run_graph(his_len, pred_len, model_type, [model_list[idx]], DatasetMap['Energy'], graph_init, run_task_name=f'{model_type}_prior_Energy_{model_list[idx]}_{graph_init}_{his_len}_{pred_len}') 
    auto_runner.run_graph(his_len, pred_len, model_type, [model_list[idx]], DatasetMap['Finance'], graph_init, run_task_name=f'{model_type}_prior_Finance_{model_list[idx]}_{graph_init}_{his_len}_{pred_len}')
    auto_runner.run_graph(his_len, pred_len, model_type, [model_list[idx]], DatasetMap['Weather'], graph_init, run_task_name=f'{model_type}_prior_Weather_{model_list[idx]}_{graph_init}_{his_len}_{pred_len}') 

if __name__ == '__main__':
    # his_len_list = [12, 16, 24, 48, 96]
    his_len_list = [96]
    graph_init_list = ['random']#, 'invers_pearson', 'random']
    pred_len = 12
    for his_len in his_len_list:
        # out_path = os.path.join('./auto_run/data_mode_2', f'{his_len}-{pred_len}')
        out_path = os.path.join(f'./Tensor_output/auto_run/{data_mode}', f'{his_len}-{pred_len}')
        ensure_dir(out_path)
        # run_all(his_len, pred_len, out_path, 1)
        # out_path = os.path.join(f'./Tensor_output/auto_run/{data_mode}', f'{his_len}-{pred_len}')
        # ensure_dir(out_path)
        # run_by_list(his_len, pred_len, out_path, 'Tensor', ['MTGNN'])
        # run_all(his_len, pred_len, f'./Tensor_output/auto_run/{data_mode}', 3)

        for graph_init in graph_init_list:
            out_path = os.path.join(f'./Tensor_output/auto_run/{data_mode}', f'{his_len}-{pred_len}')
            ensure_dir(out_path)
            run_prior_graph(his_len, pred_len, out_path, graph_init)


    # auto_runner = AutoRunner('./auto_run')
    # his_len = 24
    # pred_len = 12

    # model_type = 'Stat'
    # model_list = ModelMap[model_type]
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_Traffic_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_Natrual_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_Energy_{his_len}_{pred_len}')

    # model_type = 'MultiVar'
    # model_list = ModelMap[model_type]
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_Traffic_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_Natrual_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_Energy_{his_len}_{pred_len}')
    
    # model_type = 'Tensor'
    # model_list = TensorGraphMap['none']
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_ng_Traffic_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_ng_Natrual_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_ng_Energy_{his_len}_{pred_len}')
    
    # model_type = 'Tensor'
    # model_list = TensorGraphMap['learned']
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Traffic'], run_task_name=f'{model_type}_learned_Traffic_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Natrual'], run_task_name=f'{model_type}_learned_Natrual_{his_len}_{pred_len}')
    # auto_runner.auto_run(his_len, pred_len, model_type, model_list, DatasetMap['Energy'], run_task_name=f'{model_type}_learned_Energy_{his_len}_{pred_len}')
    
    # model_type = 'Tensor'
    # graph_init = 'pearson'
    # auto_runner.run_graph(his_len, pred_len, model_type, TensorGraphMap['prior'], DatasetMap['Traffic'], graph_init=graph_init, run_task_name=f'{model_type}_{graph_init}_prior_Traffic_{his_len}_{pred_len}')
    # auto_runner.run_graph(his_len, pred_len, model_type, TensorGraphMap['prior'], DatasetMap['Natrual'], graph_init=graph_init, run_task_name=f'{model_type}_{graph_init}_Traffic_{his_len}_{pred_len}') 
    # auto_runner.run_graph(his_len, pred_len, model_type, TensorGraphMap['prior'], DatasetMap['Energy'], graph_init=graph_init, run_task_name=f'{model_type}_{graph_init}_Traffic_{his_len}_{pred_len}') 

    # basic_run_config = yaml.safe_load(open(TEMPLATE_PATH, 'r'))
    # # dataset_list = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMS20', 'PEMSBAY'] 
    # # dataset_list = ['JONAS_NYC_bike', 'JONAS_NYC_taxi', 'Metr-LA']
    # # dataset_list = ['PEMS07', 'PEMS08', 'PEMS20', 'PEMSBAY', 'ETT_hour', '']
    # model_type = 'Tensor'
    # # model_list = ['TimesNet', 'ST_Norm', 'StemGNN', 'STGCN']
    # model_list = ['TTS_Norm', 'GMRL', 'NET3', 'AGCRN', 'MTGNN']
    # # model_list = ['AGCRN', 'MTGNN']
    # # model_list = ['HM']
    # logger = 'none'
    # all_results = {}
    
    # time_stanp = time.strftime("%m-%d-%H-%M", time.localtime())
    # output_file_path = f'./log/auto_run_output_{model_type}_{time_stanp}.txt'
    # output_file = open(output_file_path, 'w')
    # for dataset_name in dataset_list:
    #     basic_run_config['project_name'] = f"TensorTS_{dataset_name}_16_3"
    #     pkl_path = os.path.join(DATASET_PATH, dataset_name)
    #     files = os.listdir(pkl_path)
    #     for file in files:
    #         if file.endswith('.pkl'):
    #             pkl_path = os.path.join(pkl_path, file)
    #             break
    #     basic_run_config['dataset_pkl'] = pkl_path
    #     for model_name in model_list:
    #         basic_run_config['model_name'] = model_name
    #         basic_run_config['model_type'] = model_type
    #         basic_run_config['logger'] = logger
    #         try:
    #             task = TensorTask(basic_run_config)
    #             task.train()
    #             result = task.test()
    #             # yaml.dump(result, output_file)
    #             output_file.write(f"{dataset_name}--{model_name}\n{result}\n")
    #             output_file.flush()
    #             all_results[f"{model_name}---{dataset_name}"] = result
    #             print('='*40)
    #             print(f"{model_name}---{dataset_name}")
    #             print(result)
    #             print('='*40)
    #         except:
    #             output_file.write(f"{dataset_name}--{model_name}\nNone\n")
    #             output_file.flush()
    #             continue
    #     output_file.write('\n')
    #     output_file.flush()
    # print(result)

      

    