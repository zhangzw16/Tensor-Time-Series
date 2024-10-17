import os
import time
import yaml
import torch
import numpy as np
from tasks.task_base import TaskBase
from models import ModelManager
from datasets.dataset import TTS_Dataset
from datasets.dataloader import TTS_DataLoader
from utils.evaluation import Evaluator
from utils.logger.Logger import LoggerManager
from utils.graph.graphGenerator import GraphGeneratorManager


class TensorTask(TaskBase):
    def __init__(self, configs:dict={}) -> None:
        super().__init__(configs)
        # load configuration
        self.init_time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        print(f"TensorTask init... --> {self.init_time_stamp}")
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
        model_manager = ModelManager()
        if self.model_type != 'Tensor':
            raise ValueError(f"model_type: {self.model_type} is not Tensor.")
        if model_manager.is_prior_graph(self.model_name):
            graph_init = f"-{configs['graph_init']}"
        else:
            graph_init = ''
        task_id = f"{self.model_name}-{self.data_mode}-{self.his_len}-{self.pred_len}{graph_init}-{normalizer_name}"
        self.output_dir = os.path.join(self.output_dir, self.project_name, task_id)
        # ensure output_dir
        self.ensure_output_dir(self.output_dir)
        with open(os.path.join(self.output_dir, 'configs.yml'), 'w') as file:
            yaml.dump(configs, file)

        # prepare for dataset
        self.dataset = TTS_Dataset(self.pkl_path, 
                                   his_len=self.his_len, pred_len=self.pred_len ,
                                   test_ratio=0.1, valid_ratio=0.1, seed=self.seed, data_mode=self.data_mode)
        self.trainloader = TTS_DataLoader(self.dataset, 'train', batch_size=self.batch_size, drop_last=False)
        self.validloader = TTS_DataLoader(self.dataset, 'valid', batch_size=self.batch_size, drop_last=False)
        self.testloader  = TTS_DataLoader(self.dataset, 'test' , batch_size=1, drop_last=False)
        print("Preparation for dataset is done.")

        # prepare for model
        print("Init model and logger...")
        model_configs = configs.copy()
        model_configs['normalizer'] = self.dataset.get_normalizer(norm=normalizer_name)
        graph_init = model_configs['graph_init']
        model_configs['graphGenerator'] = GraphGeneratorManager(graph_init, self.dataset)
        model_configs['tensor_shape'] = self.dataset.get_tensor_shape()
        self.model = model_manager.get_model_class(self.model_name)(model_configs)
        self.model.set_device(self.device)
        print(f"Preparation for model ({self.model_type}, {self.model_name}) is done.")

        # prepare for evaluation
        self.eval_verbose  = configs['evaluator_verbose']
        self.metrics_list  = configs['metrics_list']
        self.metrics_thres = configs['metrics_thres']
        self.evaluator = Evaluator(self.metrics_list, self.metrics_thres)
        print("Preparation for evaluation is done.")

        # logger
        if self.configs['mode'] == 'train':
            logger_manager = LoggerManager()
            run_name = f"{self.model_name}-{self.data_mode}-{self.his_len}-{self.pred_len}-{normalizer_name}"
            self.logger = logger_manager.init_logger(self.logger_name, self.output_dir, self.project_name, run_name, self.configs)
            self.logger.init()
            print(f"Preparation for logger ({self.logger_name}) is done.")

        # basic info:
        print('-'*40)
        print('Task Infomation:')
        print(f"Model: {self.model_name}, Type: {self.model_type}")
        print(f"Logger: {self.logger_name}, Project: {self.project_name}")
        print(f"Dataset: {self.pkl_path}")
        print(f"Data shape: {self.dataset.get_data_shape()}")
        print(f"his_len: {self.his_len}, pred_len: {self.pred_len}, normalizer: {normalizer_name}")
        print(f"max_epoch: {self.max_epoch}, early_stop: {self.early_stop_max}")
        print(f"The output path: {self.output_dir}")
        print('-'*40)
        
    def train(self):
        self.best_epoch_info = {}
        for i in range(self.max_epoch):
            epoch_info = {}
            epoch_mean_train_loss = self.epoch_train()
            epoch_mean_valid_loss, valid_result = self.epoch_valid()
            print(f"epoch: {i}, mean_train_loss: {epoch_mean_train_loss:.3f}, mean_valid_loss:{epoch_mean_valid_loss:.3f}")
            # logger info
            epoch_info['train/loss'] = epoch_mean_train_loss
            epoch_info['valid/loss'] = epoch_mean_valid_loss
            for metric in valid_result:
                epoch_info[f'valid/{metric}'] = valid_result[metric]
            self.logger.log(epoch_info)
            early_stop_flag = self.early_stop(epoch_mean_valid_loss, epoch_info)
            # early stop
            if early_stop_flag:
                break
        self.logger.close()
        # save_path = os.path.join(self.output_dir, 'model.pth')
        # self.model.save_model(save_path)
        # print(f'model saved in: {save_path}')


        # TODO: show training summary
        print('training finished...')
        print(f'The best valid loss: {self.best_valid_loss}')
        if self.best_epoch_info is not None:
            print('='*40)
            for key in self.best_epoch_info:
                print(f"{key}: {self.best_epoch_info[key]}")
            print('='*40)

    def epoch_train(self):
        self.model.train()
        loss_list = []
        for seq, aux_info in self.trainloader.get_batch(separate=False):
            # print(f"train: {seq.shape}");exit()
            seq = seq.to(self.device)
            pred, truth = self.model.forward(seq, aux_info)
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
            for seq, aux_info in self.validloader.get_batch(separate=False):
                seq = seq.to(self.device)
                pred, truth = self.model.forward(seq, aux_info)
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
        # load model
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join(self.output_dir, 'model.pth')
            if not os.path.exists(self.model_path):
                raise FileExistsError(f"can not find .pth file... {self.model_path}")
        print(f'load model from {self.model_path}')
        self.model.load_model(self.model_path)
        # eval mode
        self.model.eval()
        with torch.no_grad():
            pred_list = []
            truth_list = []
            hist_list = []
            for seq, aux_info in self.testloader.get_batch(separate=False):
                # print(f'seq: {seq.shape}')
                seq = seq.to(self.device)
                hist = seq[:, :self.his_len, :, :].cpu().numpy()
                pred, truth = self.model.forward(seq, aux_info)
                pred = pred.cpu().numpy()
                truth = truth.cpu().numpy()
                pred_list.append(pred)
                truth_list.append(truth)
                hist_list.append(hist)
                # result = self.evaluator.eval(pred, truth, verbose=self.eval_verbose)
                # print(result)
        pred_list = np.array(pred_list).squeeze()
        truth_list = np.array(truth_list).squeeze()
        hist_list = np.array(hist_list).squeeze()
        result = self.evaluator.eval(pred_list, truth_list, verbose=self.eval_verbose)
        # add scaled result evaluation
        scaled_result = self.evaluator.scaled_eval(hist_list, pred_list, truth_list, verbose=self.eval_verbose)
        result.update(scaled_result)
        print(result)
        return result
    # test different input/output for model
    # def test_model_io_shape(self):
    #     self.model.train()
    #     for seq, aux_info in self.trainloader.get_batch(separate=False):
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