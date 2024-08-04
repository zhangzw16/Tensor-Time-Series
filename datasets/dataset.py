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
        # self.data = self.data_pkl['data']
        # data_shape = self.data.shape
        # self.time_range = data_shape[0]
        # self.dim1_range = data_shape[1]
        # self.dim2_range = data_shape[2]
        # self.sample = 1
        # split data
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        data_index = list(range(int(self.time_range)-(his_len+pred_len)))
        # random.shuffle(data_index)
        train_index_end = int(self.time_range*train_ratio)
        valid_index_end = int(self.time_range*valid_ratio) + train_index_end
        self.trainset = data_index[:train_index_end]
        random.shuffle(self.trainset)
        self.validset = data_index[train_index_end:valid_index_end]
        random.shuffle(self.validset)
        self.testset  = data_index[valid_index_end:]
        random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }
        # print(len(self.trainset), len(self.validset), len(self.testset));exit()

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
            self.time_range = data_shape[0]
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]
        elif data_mode == 2:
            self.data = self.data.reshape(self.data.shape[0], -1, 1)
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]

    def get_tensor_shape(self):
        return (self.dim1_range, self.dim2_range)
    
    def get_data_shape(self):
        return (self.time_range, self.dim1_range, self.dim2_range)

    def get_normalizer(self, norm='none'):
        if norm == 'none':
            return DoNothing()
        elif norm == 'std':
            train_range = int(self.time_range * self.train_ratio)
            train_data = self.data[:train_range]
            # normalization and inverse
            scaler = StandNormalizer(mean=np.mean(train_data), std=np.std(train_data))
            return scaler
        else:
            raise ValueError(f'unknown normalizer: {norm}...')

'''
Name: Multivar Dataset:
Dataset format: 
(time_series_num, time_range, dim_range)
    - time_series_num: number of time series
    - time_range: number of time steps
    - dim_range: number of dimensions
'''
class MTS_Dataset:
    def __init__(self, pkl_path:str, his_len:int, pred_len:int,
                 test_ratio=0.1, valid_ratio=0.1, seed:int=2024, data_mode:int=0) -> None:
        random.seed(seed)
        self.his_len = his_len
        self.pred_len = pred_len
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        # set data mode
        self.set_data_mode(data_mode)
        # split dataset
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        # default step = 1
        # split:
        # | <--- trian ---> | <- valid ->  | <- test -> |
        data_index = list(range(int(self.time_range)-(his_len+pred_len)))
        train_index_end = int(self.time_range*train_ratio)
        valid_index_end = int(self.time_range*valid_ratio) + train_index_end
        self.trainset = data_index[:train_index_end]
        self.validset = data_index[train_index_end:valid_index_end]
        self.testset  = data_index[valid_index_end:]
        # random shuffle the dataset
        random.shuffle(self.trainset)
        random.shuffle(self.validset)
        random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }

    def set_data_mode(self, data_mode:int):
        self.data = self.data_pkl['data']
        # raw data shape: (time, dim1, dim2)
        # set data mode to change the shape
        # - 0: (time, dim1*dim2, 1)
        # - 1: (time, dim1, 1) * dim2
        # - 2: (time, dim2, 1) * dim1
        if data_mode == 0:
            self.data = self.data.reshape(1, self.data.shape[0], -1, 1)
            data_shape = self.data.shape
            self.time_series_num = int(data_shape[0])
            self.time_range = int(data_shape[1])
            self.dim_range = int(data_shape[2])
        elif data_mode == 1:
            self.data = self.data.transpose(2,0,1)
            self.data = np.expand_dims(self.data, axis=-1)
            data_shape = self.data.shape
            self.time_series_num = int(data_shape[0])
            self.time_range = int(data_shape[1])
            self.dim_range = int(data_shape[2])
        elif data_mode == 2:
            self.data = self.data.transpose(1,0,2)
            self.data = np.expand_dims(self.data, axis=-1)
            data_shape = self.data.shape
            self.time_series_num = int(data_shape[0])
            self.time_range = int(data_shape[1])
            self.dim_range = int(data_shape[2])

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_time_series_num(self):
        return self.time_series_num

    def get_dim_num(self):
        return self.dim_range
    
    def get_data_shape(self):
        return (self.get_time_series_num, self.time_range, self.dim_range)

    def get_his_pred_from_idx(self, time_series_idx:int, idx:int):
        if time_series_idx >= self.time_series_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.time_series_num}")
        his = self.data[time_series_idx, idx: int(idx+self.his_len)]
        pred = self.data[time_series_idx, int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        return his, pred
    
    def get_seq_from_idx(self, time_series_idx:int ,idx:int):
        if time_series_idx >= self.time_series_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.time_series_num}")
        win = int(self.his_len + self.pred_len)
        data = self.data[time_series_idx, idx: idx+win]
        return data
    
    def get_normalizer(self, norm='none')->list:
        if norm == 'none':
            normalizer_list = []
            for i in range(self.time_series_num):
                normalizer_list.append(DoNothing())
            return normalizer_list
        elif norm == 'std':
            normalizer_list = []
            train_range = int(self.time_range * self.train_ratio)
            train_data  = self.data[:, :train_range]
            for i in range(self.time_series_num):
                train_data_i = train_data[i]
                scaler = StandNormalizer(mean=np.mean(train_data_i), std=np.std(train_data_i))
                normalizer_list.append(scaler)
            return normalizer_list
        else:
            raise ValueError(f'unknown normalizer: {norm}...')