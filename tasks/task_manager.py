import os
import time
import yaml
from models import ModelManager
from tasks.tensor_task import TensorTask
from tasks.multivar_task import MultivarTask

# Basic Configurations
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = {
    'Tensor': os.path.join(CURRENT_PATH, 'tensor_tasks_template.yaml'),
    'Multivar': os.path.join(CURRENT_PATH, 'multivariate_tasks_template.yaml'),
}
DATASET_PATH = '/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/datasets/data'
'''
Name: TaskManager
Info: start a task with given configs automatically.
'''
class TaskManager:
    def __init__(self, output_dir:str) -> None:
        self.template_path = TEMPLATE_PATH
        self.output_dir = output_dir
        self.model_manager = ModelManager()
        self.task_map = {
            'Tensor': self.TensorTaskRun,
            'MultiVar': self.MultiVarTaskRun,
        }

    def TaskRun(self, dataset_name:str, model_name:str, configs:dict, only_test:bool=False):
        model_type = self.model_manager.get_model_type(model_name)
        result = self.task_map[model_type](dataset_name, model_name, configs, only_test)
        return result

    # search .pkl file accroding to dataset_name
    def search_pkl(self, dataset_name:str):
        pkl_path = os.path.join(DATASET_PATH, dataset_name)
        files = os.listdir(pkl_path)
        for file in files:
            if file.endswith('.pkl'):
                return os.path.join(pkl_path, file)
        raise ValueError(f"Can not find .pkl file in {pkl_path}")
    # format results to float
    def format_result(self, result:dict):
        for key in result.keys():
            result[key] = float(result[key])
        return result
    
    # Task for Tensor Model
    def TensorTaskRun(self, dataset_name:str, model_name:str, configs:dict, only_test:bool=False):
        task_config = configs.copy()
        task_config['dataset_pkl'] = self.search_pkl(dataset_name)
        task_config['model_name'] = model_name
        task_config['model_type'] = 'Tensor'
        try:
            task = TensorTask(task_config)
            if not only_test:
                task.train()
            result = task.test()
            result = self.format_result(result)
        except Exception as exp:
            result = str(exp)
        return result
    
    # Task for MultiVar Model
    def MultiVarTaskRun(self, dataset_name:str, model_name:str, configs:dict, only_test:bool=False):
        task_config = configs.copy()
        task_config['dataset_pkl'] = self.search_pkl(dataset_name)
        task_config['model_name'] = model_name
        task_config['model_type'] = 'MultiVar'
        try:
            task = MultivarTask(task_config)
            if not only_test:
                task.train()
            result = task.test()
            result = self.format_result(result)
        except Exception as exp:
            result = str(exp)
        return result
    

