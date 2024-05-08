import os
import pickle as pkl
import numpy as np
import random

from .normalizer import StandNormalizer, DoNothing

'''
Name: Tensor-Time-Series Dataset:
Param:
    -
Method:
    -
'''
class TTS_Dataset:
    def __init__(self, pkl_path:str, his_len:int, pred_len:int,
                 test_ratio=0.1, valid_ratio=0.1, seed=2024, data_mode:int=0) -> None:
        random.seed(seed)
        self.his_len = his_len
        self.pred_len = pred_len
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        # TTS format:
        # shape = (t, dim1, dim2)
        self.set_data_mode(data_mode)
        self.data = self.data_pkl['data']
        data_shape = self.data.shape
        self.time_range = data_shape[0]
        self.dim1_range = data_shape[1]
        self.dim2_range = data_shape[2]
        self.sample = 1
        # split data
        train_ratio = 1 - test_ratio - valid_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        data_index = list(range(int(self.time_range)-(his_len+pred_len)))
        random.shuffle(data_index)
        train_index_end = int(self.time_range*train_ratio)
        valid_index_end = int(self.time_range*valid_ratio) + train_index_end
        self.trainset = data_index[:train_index_end]
        self.validset = data_index[train_index_end:valid_index_end]
        self.testset  = data_index[valid_index_end:]
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_his_pred_from_idx(self, idx:int):
        # TODO: check if idx is valid
        his = self.data[idx: int(idx+self.his_len)]
        pred = self.data[int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        return his, pred
    
    def get_seq_from_idx(self, idx:int):
        win = int(self.his_len + self.pred_len)
        data = self.data[idx: idx+win]
        return data

    def set_data_mode(self, data_mode:int):
        self.data = self.data_pkl['data']
        # set data mode to change the shape
        # - 0: (time, dim1, dim2)
        # - 1: (time, dim2, dim1)
        # - 2: (time, dim1*dim2, 1) 
        if data_mode == 0:
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]
        elif data_mode == 1:
            self.data = self.data.transpose(0,2,1)
            data_shape = self.data.shape
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]
        elif data_mode == 2:
            self.data = self.data.reshape(self.data.shape[0], -1, 1)
            data_shape = self.data.shape
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]

    def get_tensor_shape(self):
        return (self.dim1_range, self.dim2_range)
    
    def get_normalizer(self, norm='none'):
        if norm == 'none':
            return DoNothing()
        elif norm == 'std':
            data_index = self.trainset
            train_data = []
            for idx in data_index:
                data, _ = self.get_his_pred_from_idx(idx)
                train_data.append(data)
            train_data = np.array(train_data)
            # normalization and inverse
            scaler = StandNormalizer(mean=np.mean(train_data), std=np.std(train_data))
            return scaler
        else:
            raise ValueError(f'unknown normalizer: {norm}...')



