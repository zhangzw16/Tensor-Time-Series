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
from utils.autoBatch.autoBatch import AutoBatch
from utils.timer.Timer import Timer

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
        self.normalizer_name = configs['normalizer']
        self.model_type = configs['model_type']
        self.model_name = configs['model_name']

        # backup configs
        self.configs = configs.copy()
        # check model_type
        if self.model_type != 'MultiVar':
            raise ValueError(f"model_type: {self.model_type} is not MultiVar.")
               
        self.task_id = f"{self.dataset_name}-{self.model_name}-{self.his_len}-{self.pred_len}-{self.data_mode}-{self.normalizer_name}-{self.init_time_stamp}"
        self.output_dir = os.path.join(self.output_dir, self.project_name, self.task_id)
        self.ensure_output_dir(self.output_dir)
        with open(os.path.join(self.output_dir, 'configs.yml'), 'w') as file:
            yaml.dump(configs, file)
        # Init Timer
        self.timer = Timer(self.model_name, self.dataset_name)
        # prepare for dataset
        self.dataset = MTS_DatasetManager(self.pkl_path, 
                                   his_len=self.his_len, pred_len=self.pred_len, normalizer_name=self.normalizer_name,
                                   test_ratio=0.1, valid_ratio=0.1, seed=self.seed, data_mode=self.data_mode)
        self.subset_num = self.dataset.get_subset_num()
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

    def init_new_model_logger(self, subset_idx:int):
        print(f"Init model and logger... ({int(subset_idx+1)}/{self.subset_num})")
        self.run_dir = os.path.join(self.output_dir, f'run_{subset_idx}')
        self.ensure_output_dir(self.run_dir)
        # prepare for model
        model_manager = ModelManager()
        model_configs = self.configs.copy()
        model_configs['tensor_shape'] = self.dataset.get_tensor_shape()
        # model_configs['dim_num'] = self.dataset.get_modality_num()
        self.timer.mark_start_time('model_init')
        self.model = model_manager.get_model_class(self.model_name)(model_configs)
        model_init_time = self.timer.mark_end_time('model_init')
        self.timer.mark_start_time('model_set_device')
        self.model.set_device(self.device)
        model_set_device_time = self.timer.mark_end_time('model_set_device')
        print(f"Preparation for model ({self.model_type}, {self.model_name}) is done.")
        print(f"Duration >> Model Init: {model_init_time:.4f}s, Model Set Device: {model_set_device_time:.4f}s")
        # prepare for dataloader
        self.trainset = MTS_Dataset_Torch(self.dataset, 'train', subset_idx=subset_idx)
        self.validset = MTS_Dataset_Torch(self.dataset, 'valid', subset_idx=subset_idx)
        self.testset = MTS_Dataset_Torch(self.dataset, 'test', subset_idx=subset_idx)
        # AutoBatch
        print(f">>>> Batch_size: {self.batch_size}")
        if self.batch_size == 0:
            print(">>>> Batch_size is 0, search best batch size...")
            autoBatchManager = AutoBatch(self.model, self.trainset)
            self.timer.mark_start_time('auto_batch')
            best_batch_size = autoBatchManager.search_batch()
            auto_batch_time = self.timer.mark_end_time('auto_batch')
            print(f"Best BachSize: {best_batch_size} ({auto_batch_time:.4f}s)")
            self.batch_size = best_batch_size
            self.configs['batch_size'] = best_batch_size

        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.validloader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        print(f"trainset: {len(self.trainset)}, validset: {len(self.validset)}, testset: {len(self.testset)}")
        print(f"Preparation for dataloader is done.")

        # LR Finder
        if model_configs['lr_finder'] and model_configs['mode'] == 'train':
            print("LR Finder is enable")
            lr_finder_manager = LRFinder_Manager(self.model_name, model_configs, self.trainloader, self.validloader, self.run_dir, self.normalizer, self.device)
            self.timer.mark_start_time('lr_finder')
            lr_finder_manager.search_lr()
            lr_finder_time = self.timer.mark_end_time('lr_finder')
            best_mean_lr = lr_finder_manager.get_best_mean_lr()
            lr_finder_manager.save_plot()
            print(f"LR Finder is done. The best learning rate is: {best_mean_lr} ({lr_finder_time:.4f}s)")
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
        print(f"Batch_size: {self.batch_size}")
        print(f"his_len: {self.his_len}, pred_len: {self.pred_len}, normalizer: {self.normalizer_name}")
        print(f"max_epoch: {self.max_epoch}, early_stop: {self.early_stop_max}")
        print(f"The output path: {self.run_dir}")
        print(f"LR Finder: {self.configs['lr_finder']}")
        print(f"Optimizer: lr: {self.configs['lr']}, eps: {self.configs['eps']}, weight_decay: {self.configs['weight_decay']}")
        print(f"Scheduler: {self.scheduler_name}")
        print('-'*40)

    def epoch_train(self, run_idx:int=0):
        self.model.train()
        loss_list = []
        self.timer.mark_start_time('one_epoch_train')
        for seq in self.trainloader:
            # normalization
            # print(seq.size())
            # exit()
            norm_seq = seq
            norm_seq = norm_seq.to(self.device)
            # forward & get loss
            norm_pred, norm_truth = self.model.forward(norm_seq)
            epoch_train_loss = self.model.get_loss(norm_pred, norm_truth)
            # backward
            self.model.backward(epoch_train_loss)
            # record loss
            loss_list.append(epoch_train_loss.item())
        one_epoch_time = self.timer.mark_end_time('one_epoch_train')
        print(f"one valid epoch: {one_epoch_time:.4f}s")
        mean_loss = sum(loss_list) / len(loss_list)
        return mean_loss, one_epoch_time

    def epoch_valid(self, run_idx:int=0):
        self.model.eval()
        loss_list = []
        pred_list = []
        truth_list = []
        norm_pred_list = []
        norm_truth_list = []
        self.timer.mark_start_time('one_epoch_valid')
        with torch.no_grad():
            for seq in self.validloader:
                # normalization
                norm_seq = seq
                norm_seq = norm_seq.to(self.device)
                # forward & get loss
                norm_pred, norm_truth = self.model.forward(norm_seq)
                epoch_valid_loss = self.model.get_loss(norm_pred, norm_truth)
                loss_list.append(epoch_valid_loss.item())
                # pred & truth
                norm_pred = norm_pred.cpu().detach().numpy()
                norm_truth = norm_truth.cpu().detach().numpy()
                norm_pred_list.append(norm_pred)
                norm_truth_list.append(norm_truth)
                # inverse normalization
                # pred = self.normalizer.inverse_transform(norm_pred)
                # truth = self.normalizer.inverse_transform(norm_truth)
                # pred_list.append(pred)
                # truth_list.append(truth)
        one_epoch_time = self.timer.mark_end_time('one_epoch_valid')
        print(f"one valid epoch: {one_epoch_time:.4f}s")
        mean_loss = sum(loss_list) / len(loss_list)
        # evaluation
        # pred = np.concatenate(pred_list, axis=0)
        # truth = np.concatenate(truth_list, axis=0)
        norm_pred = np.concatenate(norm_pred_list, axis=0)
        norm_truth = np.concatenate(norm_truth_list, axis=0)
        # result = self.evaluator.eval(pred, truth, verbose=self.eval_verbose)
        norm_result = self.evaluator.eval(norm_pred, norm_truth, verbose=self.eval_verbose)
        res = {
            # 'res': result,
            'res': {},
            'norm_res': norm_result
        }
        print(f"Valid result:\n{res}")
        return mean_loss, res, one_epoch_time

    def train(self, subset_list:list=[]):
        if subset_list == []:
            subset_list = list(range(self.subset_num))
        for subset_idx in subset_list:
            # reset best_valid_loss
            self.best_valid_loss = math.inf
            self.early_stop_cnt = 0
            # init model and logger
            self.init_new_model_logger(subset_idx)
            self.best_epoch_info = {}
            for i in range(self.max_epoch):
                epoch_info = {
                    'epoch': i,
                }
                epoch_mean_train_loss, one_epoch_time_train = self.epoch_train(subset_idx)
                epoch_mean_valid_loss, valid_result, one_epoch_time_valid = self.epoch_valid(subset_idx)
                # scheduler
                if self.scheduler_name == 'ReduceLROnPlateau':
                    self.scheduler.step(epoch_mean_valid_loss)
                else:
                    self.scheduler.step()
                print(f"epoch: {i}, mean_train_loss: {epoch_mean_train_loss:.3f}, mean_valid_loss:{epoch_mean_valid_loss:.3f}")
                # logger info
                # train/
                epoch_info['train/loss'] = epoch_mean_train_loss
                epoch_info['train/one_epoch_time'] = one_epoch_time_train
                epoch_info['train/learning_rate'] = self.model.optim.param_groups[0]['lr']
                print(f"learning rate: {epoch_info['train/learning_rate']}")
                # valid/
                epoch_info['valid/loss'] = epoch_mean_valid_loss
                epoch_info['valid/one_epoch_time'] = one_epoch_time_valid
                for metric in valid_result:
                    epoch_info[f'valid/{metric}'] = valid_result[metric]
                # log info
                self.logger.log(epoch_info)
                # save checkpoint
                self.save_checkpoint(i, save_dir=self.run_dir)
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
        self.configs['mode'] = 'test'
        # if model_path is .pth file
        if self.model_path.endswith('.pth'):
            run_name = os.path.basename(os.path.dirname(self.model_path))
            run_idx = int(run_name.split('_')[-1])
            idx_list = [run_idx]
            
        # specify the idx_list
        if idx_list == []:
            idx_list = list(range(self.subset_num))

        test_result = {}
        self.configs['mode'] = 'test' 
        for run_idx in idx_list:
            self.init_new_model_logger(run_idx)
            test_result[f'run_{run_idx}'] = {}
            if self.model_path == '':
                run_dir = os.path.join(self.output_dir, f'run_{run_idx}')
                trained_model_path = os.path.join(run_dir, 'model.pth')
            else:
                trained_model_path = self.model_path
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
                norm_pred_list = []
                norm_truth_list = []
                norm_hist_list = []
                for seq in self.testloader:
                    # normalization
                    norm_seq = seq
                    norm_hist = norm_seq[:, :self.his_len, :, :].numpy()
                    norm_seq = norm_seq.to(self.device)
                    # forward (inference)
                    norm_pred, norm_truth = self.model.forward(norm_seq)
                    # evaluation
                    norm_pred = norm_pred.cpu().detach().numpy()
                    norm_truth = norm_truth.cpu().detach().numpy()
                    norm_pred_list.append(norm_pred)
                    norm_truth_list.append(norm_truth)
                    norm_hist_list.append(norm_hist)
                    # inverse normalization
                    # pred = self.normalizer.inverse_transform(norm_pred)
                    # truth = self.normalizer.inverse_transform(norm_truth)
                    # hist = self.normalizer.inverse_transform(norm_hist)
                    # pred_list.append(pred)
                    # truth_list.append(truth)
                    # hist_list.append(hist)
            # evaluation
            # pred_list = np.concatenate(pred_list, axis=0)
            # truth_list = np.concatenate(truth_list, axis=0)
            # hist_list = np.concatenate(hist_list, axis=0)
            norm_pred_list = np.concatenate(norm_pred_list, axis=0)
            norm_truth_list = np.concatenate(norm_truth_list, axis=0)
            norm_hist_list = np.concatenate(norm_hist_list, axis=0)
            # results
            # result = self.evaluator.eval(pred_list, truth_list, verbose=self.eval_verbose)
            # scaled_result = self.evaluator.scaled_eval(hist_list, pred_list, truth_list, verbose=self.eval_verbose)
            # result.update(scaled_result)
            # norm_result
            norm_result = self.evaluator.eval(norm_pred_list, norm_truth_list, verbose=self.eval_verbose)
            norm_scaled_result = self.evaluator.scaled_eval(norm_hist_list, norm_pred_list, norm_truth_list, verbose=self.eval_verbose)
            norm_result.update(norm_scaled_result)
            res = {
                # 'res': result,
                'res': {},
                'norm_res': norm_result
            }
            print(f"Test result:\n{res}")
            test_result[f'run_{run_idx}'] = res
        return test_result