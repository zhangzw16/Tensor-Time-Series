import os
import yaml
import torch
import numpy as np
from tasks.task_base import TaskBase
from models import ModelManager
from models.model_base import TensorModelBase
from datasets.dataset import TTS_Dataset
from datasets.dataloader import TTS_DataLoader
from utils.evaluation import Evaluator


class TensorTask(TaskBase):
    def __init__(self, configs:dict={}) -> None:
        super().__init__(configs)
        # load configuration
        self.seed = configs['seed']
        self.device = configs['task_device']
        self.log_dir = configs['log_dir']
        self.model_save_dir = configs['model_save_dir']
        self.batch_size = configs['batch_size']
        self.max_epoch = configs['max_epoch']
        self.early_stop_max = configs['early_stop_max']
        self.early_stop_cnt = 0

        self.pkl_path = configs['dataset_pkl']
        self.his_len = configs['his_len']
        self.pred_len = configs['pred_len']

        # prepare for model
        model_manager = ModelManager()
        self.model_name = configs['model_name']
        model_configs = {}
        self.model = model_manager.get_model_class(self.model_name)(model_configs)
        self.model.set_device(self.device)

        # prepare for dataset
        self.dataset = TTS_Dataset(self.pkl_path, 
                                   his_len=self.his_len, pred_len=self.pred_len ,test_ratio=0.1, valid_ratio=0.1, seed=2024)
        self.trainloader = TTS_DataLoader(self.dataset, 'train', batch_size=16, drop_last=False)
        self.validloader = TTS_DataLoader(self.dataset, 'valid', batch_size=16, drop_last=False)
        self.testloader = TTS_DataLoader(self.dataset, 'test', batch_size=1, drop_last=False)
        
        # prepare for evaluation
        self.eval_verbose  = configs['evaluator_verbose']
        self.metrics_list  = configs['metrics_list']
        self.metrics_thres = configs['metrics_thres']
        self.evaluator = Evaluator(self.metrics_list, self.metrics_thres)

    def train(self):
        for i in range(self.max_epoch):
            epoch_mean_train_loss = self.epoch_train()
            epoch_mean_valid_loss, valid_result = self.epoch_valid()
            early_stop_flag = self.early_stop(epoch_mean_valid_loss)
            print(f"epoch: {i}, mean_train_loss: {epoch_mean_train_loss:.3f}, mean_valid_loss:{epoch_mean_valid_loss:.3f}")
            if early_stop_flag:
                # TODO: save model ...
                break
        # TODO: show training summary
        print('training finished...')

    def epoch_train(self):
        self.model.train()
        loss_list = []
        for seq in self.trainloader.get_batch(separate=False):
            seq = seq.to(self.device)
            pred, truth = self.model.forward(seq)
            epoch_train_loss = self.model.get_loss(pred, truth)
            self.model.backward(epoch_train_loss)
            loss_list.append(epoch_train_loss.item())
        mean_loss = sum(loss_list)/len(loss_list)
        return mean_loss
    
    def epoch_valid(self):
        self.model.eval()
        loss_list = []
        pred_list = []
        truth_list = []
        with torch.no_grad():
            for seq in self.validloader.get_batch(separate=False):
                seq = seq.to(self.device)
                pred, truth = self.model.forward(seq)
                epoch_valid_loss = self.model.get_loss(pred, truth)
                loss_list.append(epoch_valid_loss.item())
                pred = pred.cpu().numpy()
                truth = truth.cpu().numpy()
                pred_list.append(pred)
                truth_list.append(truth)
        mean_loss = sum(loss_list)/len(loss_list)
        pred = np.array(pred_list).squeeze()
        truth = np.array(truth_list).squeeze()
        result = self.evaluator.eval(pred, truth, verbose=self.eval_verbose)
        return mean_loss, result

    def test(self):
        self.model.eval()
        pred_list = []
        truth_list = []
        with torch.no_grad():
            for seq in self.testloader.get_batch(separate=False):
                seq = seq.to(self.device)
                pred, truth = self.model.forward(seq)
                pred_list.append(pred)
                truth_list.append(truth)
        pred = np.array(pred_list)
        truth = np.array(truth_list)
        result = self.evaluator.eval(pred, truth, verbose=self.eval_verbose)
        return result