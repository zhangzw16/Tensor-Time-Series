import os
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import gc
from tasks.task_base import TaskBase
from models import ModelManager
from datasets.dataset import Stat_DatasetManager
from datasets.dataloader_torch import Stat_Dataset_Torch
from utils.evaluation import Evaluator

class StatTask(TaskBase):
    def __init__(self, config:dict={}) -> None:
        super().__init__(config)
        self.init_time_stamp = config['timestamp']
        print(f"Init Time Stamp: {self.init_time_stamp}")
        print(f"Loading configs...")
        self.seed = config['seed']
        self.device = config['task_device']
        self.output_dir = config['output_dir']
        self.model_path = config['model_path']
        self.batch_size = config['batch_size']
        self.max_epoch = config['max_epoch']
        # logger
        self.logger_name = config['logger']
        self.project_name = config['project_name']
        # dataset
        self.dataset_name = config['dataset_name']
        self.pkl_path = config['dataset_pkl']
        self.data_mode = 0
        self.his_len = config['his_len']
        self.pred_len = config['pred_len']
        self.lag_input = []
        self.normalizer_name = config['normalizer']
        self.model_type = config['model_type']
        self.model_name = config['model_name']
        
        model_manager = ModelManager()
        task_id = f"{self.dataset_name}-{self.model_name}-{self.data_mode}-{self.his_len}-{self.pred_len}-{self.normalizer_name}"
        self.output_dir = os.path.join(self.output_dir, self.project_name, task_id)
        self.dataset = Stat_DatasetManager(self.pkl_path, self.his_len, self.pred_len, normalizer_name=self.normalizer_name, 
                                           test_ratio=0.1, valid_ratio=0.1, data_mode=self.data_mode, lag_input=self.lag_input, seed=self.seed)
        config['tensor_shape'] = self.dataset.get_tensor_shape()
        self.testset = Stat_Dataset_Torch(self.dataset, 'test')
        
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        model_configs = config.copy()
        self.model = model_manager.get_model_class(self.model_name)(model_configs)
        self.model.set_device(self.device)

        # prepare for evaluation
        self.eval_verbose = config['evaluator_verbose']
        self.metrics_list = config['metrics_list']
        self.metrics_thres = config['metrics_thres']
        self.evaluator = Evaluator(self.metrics_list, self.metrics_thres)

    def train(self):
        return 
    
    def test(self):
        norm_pred_list = []
        norm_truth_list = []
        norm_hist_list = []
        for seq in self.testloader:
            norm_seq = seq
            norm_hist = norm_seq[:, :self.his_len, :, :].numpy()
            norm_seq = norm_seq.to(self.device)
            norm_pred, norm_truth = self.model.forward(norm_seq)
            norm_pred = norm_pred.cpu().detach().numpy()
            norm_truth = norm_truth.cpu().detach().numpy()
            # print(f"norm_pred: {norm_pred.shape}, norm_truth: {norm_truth.shape}, norm_hist: {norm_hist.shape}")
            # length = norm_hist.shape[1]
            # norm_pred = norm_pred[:, :length, :, :]
            # norm_truth = norm_truth[:, :length, :, :]
            norm_pred_list.append(norm_pred)
            norm_truth_list.append(norm_truth)
            norm_hist_list.append(norm_hist)
        norm_pred_list = np.concatenate(norm_pred_list, axis=0)
        norm_truth_list = np.concatenate(norm_truth_list, axis=0)
        norm_hist_list = np.concatenate(norm_hist_list, axis=0)
        norm_result = self.evaluator.eval(norm_pred_list, norm_truth_list, verbose=self.eval_verbose)
        # norm_scaled_result = self.evaluator.scaled_eval(norm_pred_list, norm_truth_list, norm_hist_list, verbose=self.eval_verbose)
        # norm_result.update(norm_scaled_result)
        res = {
            'res': {},
            'norm_res': norm_result
        }
        print(f"Test Result: {res}")
        return res