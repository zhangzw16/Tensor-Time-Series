import os
import yaml
import torch
import numpy as np

from tasks.task_base import TaskBase
from models import ModelManager

from datasets.dataset import TTS_Dataset
from datasets.dataloader import TTS_DataLoader

from utils.evaluation import Evaluator
from utils.logger.Logger import LoggerManager
from utils.graph.graphGenerator import GraphGenerator

class MultiVarTask(TaskBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        
        self.seed = configs['seed']
        self.device = configs['task_device']
        self.output_dir = configs['output_dir']
        self.model_path = configs['model_path']
        self.batch_size = configs['batch_size']
        self.max_epoch = configs['max_epoch']
        self.early_stop_max = configs['early_stop_max']
        self.early_stop_cnt = 0

        self.logger_name = configs['logger']
        self.project_name = configs['project_name']

        self.pkl_path = configs['dataset_pkl']
        self.data_mode = configs['data_mode']
        self.his_len = configs['his_len']
        self.pred_len = configs['pred_len']

        # prepare for dataset
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
