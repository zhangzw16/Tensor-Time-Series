import torch
import os
import random
import pickle
import numpy as np
from torch.utils.data import Dataset

# from datasets.dataset_manager import DatasetManager

class NET3_dataset(Dataset):
    def __init__(self, mode='train') -> None:
        self.configs = self.defualt_configs()
        self.mode = mode
        # load values
        self.pkl_path = '/home/zhuangjiaxin/workspace/Tensor-Time-Series/repos/NET3/dataset_processed/future'
        self.values  = pickle.load(open(os.path.join(self.pkl_path, 'values.pkl'), 'rb'))
        self.max_step = self.values.shape[-1]
        # if mode == 'train':
        #     self.indicators = pickle.load(open(os.path.join(self.pkl_path, 'train_idx.pkl'), 'rb'))
        # elif mode == 'valid':
        #     self.indicators  = pickle.load(open(os.path.join(self.pkl_path, 'val_idx.pkl'), 'rb'))    
        # elif mode == 'test':
        #     self.indicators  = pickle.load(open(os.path.join(self.pkl_path, 'test_idx.pkl'), 'rb'))
        #     self.configs['stride'] = 1
        
        
        self.indicators_train = pickle.load(open(os.path.join(self.pkl_path, 'train_idx.pkl'), 'rb'))
        
        self.indicators_eval  = pickle.load(open(os.path.join(self.pkl_path, 'val_idx.pkl'), 'rb'))    
        if mode == 'test':
            self.indicators_eval  = pickle.load(open(os.path.join(self.pkl_path, 'test_idx.pkl'), 'rb'))
            self.configs['stride'] = 1

        self.n_iter = int(np.ceil((self.max_step - self.configs['window_size'])/self.configs['stride']))+1 
        
        self.shapes = list(self.values.shape)[:-1]
        self._pad()
        self.starts = [i * self.configs['stride'] for i in range(self.n_iter)]
        self.ends = [i + self.configs['window_size'] for i in self.starts]
        self.start_ends = list(zip(self.starts, self.ends))

    def __len__(self):
        return self.n_iter
    
    def __getitem__(self, index):
        start, end = self.start_ends[index]
        value = self.values[..., start:end]
        indicators_train = self.indicators_train[..., start:end]
        indicators_eval = self.indicators_eval[..., start:end]
        # to tensor
        value = torch.from_numpy(value).float()
        indicators_train = torch.from_numpy(indicators_train).float()
        indicators_eval = torch.from_numpy(indicators_eval).float()
        # return
        return value, indicators_train, indicators_eval

    def defualt_configs(self):
        return {
            "window_size": 6,   # 5 (historical) + 1 (future)
            "batch_size": 100,
            "stride": 1,
            "network_name": None
        }
    
    def _pad(self):
        max_step = self.n_iter * self.configs["stride"] + self.configs["window_size"]
        diff = max_step - self.max_step
        self.shapes.append(diff)
        pad_tensor = np.zeros(self.shapes)
        self.values = np.concatenate([self.values, pad_tensor], axis=-1)
        self.indicators_train = np.concatenate([self.indicators_train, pad_tensor], axis=-1)
        self.indicators_eval = np.concatenate([self.indicators_eval, pad_tensor], axis=-1)

    def get_networks(self):
        networks = pickle.load(open(
            os.path.join(self.pkl_path, 'networks.pkl'), "rb"))
        for n in networks:
            networks[n] = torch.from_numpy(networks[n]).float()
        return networks    