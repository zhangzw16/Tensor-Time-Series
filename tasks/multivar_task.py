import os
import math
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from tasks.task_base import TaskBase
from models import ModelManager
from datasets.dataset import MTS_DatasetManager
from datasets.dataloader import MTS_DataLoader
from datasets.dataloader_torch import MTS_Dataset_Torch
from utils.evaluation import Evaluator
from utils.logger.Logger import LoggerManager
from utils.scheduler.schedulerManager import SchedulerManager
from utils.lrFinder.LR_Finder import LRFinder_Manager

class MultivarTask(TaskBase):
    def __init__(self, configs:dict={}) -> None:
        super().__init__(configs)
        # load configuration
        self.init_time_stamp = configs['timestamp']
        print(f"MultivarTask init... --> {self.init_time_stamp}")
        print(f"Task mode: {configs['mode']}")
        print(f"Loading configs...")
        self.seed = configs['seed']
        self.device = configs['task_device']
        self.output_dir = configs['output_dir']
        self.model_path = configs['model_path']
        self.batch_size = configs['batch_size']
        self.max_epoch = configs['max_epoch']
        self.early_stop_max = configs['early_stop_max']
        self.early_stop_cnt = 0
        # project & logger
        self.logger_name = configs['logger']
        self.project_name = configs['project_name']
        # dataset
        self.dataset_name = configs['dataset_name']
        self.pkl_path = configs['dataset_pkl']
        self.data_mode = configs['data_mode']
        self.his_len = configs['his_len']
        self.pred_len = configs['pred_len']
        normalizer_name = configs['normalizer']
        self.model_type = configs['model_type']
        self.model_name = configs['model_name']

        # backup configs
        self.configs = configs.copy()
        # check model_type
        if self.model_type != 'MultiVar':
            raise ValueError(f"model_type: {self.model_type} is not MultiVar.")
               
        self.task_id = f"{self.dataset_name}-{self.model_name}-{self.his_len}-{self.pred_len}-{self.data_mode}-{normalizer_name}-{self.init_time_stamp}"
        self.output_dir = os.path.join(self.output_dir, self.project_name, self.task_id)
        self.ensure_output_dir(self.output_dir)
        with open(os.path.join(self.output_dir, 'configs.yml'), 'w') as file:
            yaml.dump(configs, file)
        # prepare for dataset
        self.dataset = MTS_DatasetManager(self.pkl_path, 
                                   his_len=self.his_len, pred_len=self.pred_len ,
                                   test_ratio=0.1, valid_ratio=0.1, seed=self.seed, data_mode=self.data_mode)
        self.time_series_num = self.dataset.get_time_series_num()
        # self.trainloader = MTS_DataLoader(self.dataset, 'train', batch_size=self.batch_size, drop_last=False)
        # self.validloader = MTS_DataLoader(self.dataset, 'valid', batch_size=self.batch_size, drop_last=False)
        # self.testloader = MTS_DataLoader(self.dataset, 'test', batch_size=self.batch_size, drop_last=False)
        print(f"Preparation for dataset is done.")
        
        # prepare for evaluation
        self.eval_verbose = configs['evaluator_verbose']
        self.metrics_list = configs['metrics_list']
        self.metrics_thres = configs['metrics_thres']
        self.evaluator = Evaluator(self.metrics_list, self.metrics_thres)
        print(f"Preparation for evaluation is done.")

    def init_new_model_logger(self, run_idx:int):
        print(f"Init model and logger... ({int(run_idx+1)}/{self.time_series_num})")
        self.run_dir = os.path.join(self.output_dir, f'run_{run_idx}')
        self.ensure_output_dir(self.run_dir)
        # prepare for model
        model_manager = ModelManager()
        model_configs = self.configs.copy()
        normalizer_name = model_configs['normalizer']
        model_configs['tensor_shape'] = (self.dataset.get_dim_num() , 1)
        model_configs['normalizer'] = self.dataset.get_normalizer(normalizer_name)[run_idx]
        model_configs['dim_num'] = self.dataset.get_dim_num()
        self.model = model_manager.get_model_class(self.model_name)(model_configs)
        self.model.set_device(self.device)
        print(f"Preparation for model ({self.model_type}, {self.model_name}) is done.")

        # prepare for dataloader
        self.trainset = MTS_Dataset_Torch(self.dataset, 'train', ts_idx=run_idx)
        self.validset = MTS_Dataset_Torch(self.dataset, 'valid', ts_idx=run_idx)
        self.testset = MTS_Dataset_Torch(self.dataset, 'test', ts_idx=run_idx)
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.validloader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.testloader = DataLoader(self.testset, batch_size=1, shuffle=False, drop_last=False)
        print(f"trainset: {len(self.trainset)}, validset: {len(self.validset)}, testset: {len(self.testset)}")
        print(f"Preparation for dataloader is done.")

        # LR Finder
        if model_configs['lr_finder'] and model_configs['mode'] == 'train':
            print("LR Finder is enable")
            lr_finder_manager = LRFinder_Manager(self.model_name, model_configs, self.trainloader, self.validloader, self.run_dir, self.device)
            lr_finder_manager.search_lr()
            best_mean_lr = lr_finder_manager.get_best_mean_lr()
            lr_finder_manager.save_plot()
            print(f"LR Finder is done. The best learning rate is: {best_mean_lr}")
            print(f"plot is saved in {self.run_dir}/lr_finder.png")
            lr_finder_manager.set_optim_with_lr(self.model, best_mean_lr)
            self.configs['lr'] = best_mean_lr
            # for param_group in self.model.optim.param_groups:
            #     print(f"Learning rate: {param_group['lr']}")
            # exit()

        # prepare for scheduler
        self.scheduler_manager = SchedulerManager()
        self.scheduler_name = self.configs['scheduler']
        self.scheduler = self.scheduler_manager.get_scheduler(self.model.optim, self.scheduler_name)
        

        # prepare for logger
        if self.configs['mode'] == 'train':
            logger_manager = LoggerManager()
            self.logger = logger_manager.init_logger(self.logger_name, self.run_dir, self.project_name, self.task_id, self.configs)
            self.logger.init()
            print(f"Preparation for logger ({self.logger_name}) is done.")
        # Basic Info
        print('-'*40)
        print('Task Infomation:')
        print(f"Model: {self.model_name}, Type: {self.model_type}")
        print(f"Logger: {self.logger_name}, Project: {self.project_name}")
        print(f"Dataset: {self.pkl_path}")
        print(f"Data shape: {self.dataset.get_data_shape()}")
        print(f"his_len: {self.his_len}, pred_len: {self.pred_len}, normalizer: {normalizer_name}")
        print(f"max_epoch: {self.max_epoch}, early_stop: {self.early_stop_max}")
        print(f"The output path: {self.run_dir}")
        print(f"LR Finder: {self.configs['lr_finder']}")
        print(f"Optimizer: lr: {self.configs['lr']}, eps: {self.configs['eps']}, weight_decay: {self.configs['weight_decay']}")
        print(f"Scheduler: {self.scheduler_name}")
        print('-'*40)

    def epoch_train(self, run_idx:int=0):
        self.model.train()
        loss_list = []
        for seq in self.trainloader:
            # if self.trainloader.batch_size == 1:
            #     seq = seq.unsqueeze(0)
            # print(seq.shape)
            seq = seq.to(self.device)
            pred, truth = self.model.forward(seq)
            epoch_train_loss = self.model.get_loss(pred, truth)
            self.model.backward(epoch_train_loss)
            loss_list.append(epoch_train_loss.item())
        mean_loss = sum(loss_list) / len(loss_list)
        return mean_loss

    def epoch_valid(self, run_idx:int=0):
        self.model.eval()
        loss_list = []
        pred_list = []
        truth_list = []
        with torch.no_grad():
            for seq in self.validloader:
                # if self.validloader.batch_size == 1:
                #     seq = seq.unsqueeze(0)
                # print(seq.shape)
                seq = seq.to(self.device)
                pred, truth = self.model.forward(seq)
                epoch_valid_loss = self.model.get_loss(pred, truth)
                # update loss list
                loss_list.append(epoch_valid_loss.item())
                # pred & truth
                pred = pred.cpu().detach().numpy()
                truth = truth.cpu().detach().numpy()
                # update pred & truth list
                # print(f'pred: {pred.shape}, truth: {truth.shape}')
                pred_list.append(pred)
                truth_list.append(truth)
        mean_loss = sum(loss_list) / len(loss_list)
        # pred = np.array(pred_list).squeeze()
        # truth = np.array(truth_list).squeeze()
        pred = np.concatenate(pred_list, axis=0)
        truth = np.concatenate(truth_list, axis=0)
        result = self.evaluator.eval(pred, truth, verbose=self.eval_verbose)
        return mean_loss, result

    def train(self, idx_list:list=[]):
        if idx_list == []:
            idx_list = list(range(self.time_series_num))
        for run_idx in idx_list:
            # reset best_valid_loss
            self.best_valid_loss = math.inf
            self.early_stop_cnt = 0
            # init model and logger
            self.init_new_model_logger(run_idx)
            self.best_epoch_info = {}
            for i in range(self.max_epoch):
                epoch_info = {}
                epoch_mean_train_loss = self.epoch_train(run_idx)
                epoch_mean_valid_loss, valid_result = self.epoch_valid(run_idx)
                # scheduler
                if self.scheduler_name == 'ReduceLROnPlateau':
                    self.scheduler.step(epoch_mean_valid_loss)
                else:
                    self.scheduler.step()
                print(f"epoch: {i}, mean_train_loss: {epoch_mean_train_loss:.3f}, mean_valid_loss:{epoch_mean_valid_loss:.3f}")
                # logger info
                epoch_info['train/loss'] = epoch_mean_train_loss
                epoch_info['valid/loss'] = epoch_mean_valid_loss
                for metric in valid_result:
                    epoch_info[f'valid/{metric}'] = valid_result[metric]
                epoch_info['learning_rate'] = self.model.optim.param_groups[0]['lr']
                print(epoch_info['learning_rate'])
                self.logger.log(epoch_info)
                early_stop_flag = self.early_stop(i, epoch_mean_valid_loss, epoch_info, save_dir=self.run_dir)
                # early stop
                if early_stop_flag:
                    break
            self.logger.close()
            # training summary
            print('training finished...')
            print(f'The best valid loss: {self.best_valid_loss}')
            if self.best_epoch_info is not None:
                print('='*40)
                for key in self.best_epoch_info:
                    print(f"{key}: {self.best_epoch_info[key]}")
                print('='*40)

    def test(self, idx_list:list=[]):
        # if model_path is .pth file
        if self.model_path.endswith('.pth'):
            run_name = os.path.basename(os.path.dirname(self.model_path))
            run_idx = int(run_name.split('_')[-1])
            idx_list = [run_idx]
        # specify the idx_list
        if idx_list == []:
            idx_list = list(range(self.time_series_num))

        test_result = {}
        self.configs['mode'] = 'test' 
        for run_idx in idx_list:
            self.init_new_model_logger(run_idx)
            test_result[f'run_{run_idx}'] = {}
            run_dir = os.path.join(self.output_dir, f'run_{run_idx}')
            trained_model_path = os.path.join(run_dir, 'model.pth')
            if not os.path.exists(trained_model_path):
                print(f"can not find .pth file: {trained_model_path}")
                continue
            print(f'Load model from {trained_model_path}')
            self.model.load_model(trained_model_path)
            print(f'model loaded...')
            # eval mode
            self.model.eval()
            with torch.no_grad():
                pred_list = []
                truth_list = []
                hist_list = []
                for seq in self.testloader:
                    # if self.testloader.batch_size == 1:
                    #     seq = seq.unsqueeze(0)
                    # print(seq.shape)
                    seq = seq.to(self.device)
                    hist = seq[:, :self.his_len, :, :].cpu().numpy()
                    pred, truth = self.model.forward(seq)
                    pred = pred.cpu().numpy()
                    truth = truth.cpu().numpy()
                    pred_list.append(pred)
                    truth_list.append(truth)
                    hist_list.append(hist)
            pred_list = np.array(pred_list).squeeze()
            truth_list = np.array(truth_list).squeeze()
            hist_list = np.array(hist_list).squeeze()
            result = self.evaluator.eval(pred_list, truth_list, verbose=self.eval_verbose)
            scaled_result = self.evaluator.scaled_eval(hist_list, pred_list, truth_list, verbose=self.eval_verbose)
            result.update(scaled_result)
            print(f"Test result:\n{result}")
            test_result[f'run_{run_idx}'] = result
        return test_result
    
    # test diffrent input/output for model
    # def test_model_io_shape(self):
    #     run_idx = 0
    #     self.init_new_model_logger(run_idx)
    #     self.model.train()
    #     for seq, aux_info in self.trainloader.get_batch(run_idx, separate=False):
    #         seq = seq.to(self.device)
    #         pred, truth = self.model.forward(seq, aux_info)
    #         if pred.shape != truth.shape:
    #             result = f"Can not match the shape: {pred.shape} != {truth.shape}"
    #         else:
    #             result = "OK"
    #         # compute loss & backward
    #         loss = self.model.get_loss(pred, truth)
    #         self.model.backward(loss)
    #         break
    #     test_id = f"{self.model_name}-{self.his_len}-{self.pred_len}"
    #     return {'result': result, 'test_id': test_id}
        

