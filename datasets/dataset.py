import os
import pickle as pkl
import numpy as np
import random

from .normalizer import Sklearn_StandNormalizer, DoNothing, StandNormalizer

'''
Name: Tensor-Time-Series Dataset Manager:
Param:
    -
Method:
    -
'''
class TTS_DatasetManager:
    def __init__(self, pkl_path:str, his_len:int, pred_len:int, normalizer_name:str='none',
                 test_ratio=0.1, valid_ratio=0.1, seed=2024, data_mode:int=0) -> None:
        random.seed(seed)
        self.pkl_path = pkl_path
        self.his_len = his_len
        self.pred_len = pred_len
        self.normalizer_name = normalizer_name
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        # TTS format:
        # shape = (t, dim1, dim2)
        self.make_datasets(data_mode)
        # self.data = self.data_pkl['data']
        # data_shape = self.data.shape
        # self.time_range = data_shape[0]
        # self.dim1_range = data_shape[1]
        # self.dim2_range = data_shape[2]
        # self.sample = 1
        # split data
        # data_index = list(range(int(self.time_range)-(his_len+pred_len)))
        # # random.shuffle(data_index)
        # train_index_end = int(len(data_index)*train_ratio)
        # valid_index_end = int(len(data_index)*valid_ratio) + train_index_end
        # self.trainset = data_index[:train_index_end]
        # random.shuffle(self.trainset)
        # self.validset = data_index[train_index_end:valid_index_end]
        # random.shuffle(self.validset)
        # self.testset  = data_index[valid_index_end:]
        # random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }
        # print(len(self.trainset), len(self.validset), len(self.testset));exit()

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_his_pred_from_idx(self, idx:int):
        # his = self.data[idx: int(idx+self.his_len)]
        # pred = self.data[int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        data = self.data[idx]
        his = data[:self.his_len]
        pred = data[self.his_len:]
        return his, pred
    
    def get_seq_from_idx(self, idx:int):
        # win = int(self.his_len + self.pred_len)
        # data = self.data[idx: idx+win]
        data = self.data[idx]
        return data

    def make_datasets(self, data_mode:int):
        self.data = self.data_pkl['data']
        sample_length = self.his_len + self.pred_len
        sample_num = self.data.shape[0] - sample_length
        train_end = int(sample_num * self.train_ratio)
        valid_end = int(sample_num * self.valid_ratio) + train_end
        self.raw_train_data = self.data[:train_end]
        # normalize data
        self.normalizer = self.init_normalizer(self.normalizer_name, self.raw_train_data)
        self.data = self.normalizer.transform(self.data)
        # the shape of data read from .pkl is (T, N, M),
        # set data mode to change the shape
        # - 0: Tensor Direct        (T, N, M)
        # - 1: Modality-Independent (T x M, N, 1)
        # - 2: Modality-Individual  (T, N, 1) x M
        if data_mode == 0:
            # Tensor Direct
            # (T, N, M) -> (1, T, N, M)
            self.subset_num = 1
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.variable_num = data_shape[1]
            self.modality_num = data_shape[2]
            sample_list = []
            for i in range(sample_num):
                sample_list.append(self.data[i: i+sample_length])
            self.data = np.array(sample_list)
            self.data = np.expand_dims(self.data, axis=0)
            self.trainset = self.data[:, :train_end]
            self.validset = self.data[:, train_end:valid_end]
            self.testset  = self.data[:, valid_end:]
        elif data_mode == 1:
            # Modality-Independent
            # (T, N, M) -> (1, T x M, N, 1)
            self.subset_num = 1
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.variable_num = data_shape[1]
            self.modality_num = 1
            T = self.data.shape[0]
            N = self.data.shape[1]
            M = self.data.shape[2]
            trainset = []
            validset = []
            testset = []
            # trainset
            for i in range(train_end):
                for j in range(M):
                    sample = self.data[i:i+sample_length, :, j]
                    trainset.append(sample)
            # validset
            for i in range(train_end, valid_end):
                for j in range(M):
                    sample = self.data[i:i+sample_length, :, j]
                    validset.append(sample)
            # testset
            for i in range(valid_end, sample_num):
                for j in range(M):
                    sample = self.data[i:i+sample_length, :, j]
                    testset.append(sample)
            # dataset
            self.trainset = np.array(trainset)
            self.trainset = np.expand_dims(self.trainset, axis=0)
            self.trainset = np.expand_dims(self.trainset, axis=-1)
            self.validset = np.array(validset)
            self.validset = np.expand_dims(self.validset, axis=0)
            self.validset = np.expand_dims(self.validset, axis=-1)
            self.testset  = np.array(testset)
            self.testset  = np.expand_dims(self.testset, axis=0)
            self.testset = np.expand_dims(self.testset, axis=-1)
        elif data_mode == 2:
            # Modality-Individual
            # (T, N, M) -> (M, T, N, 1)
            self.subset_num = int(self.data.shape[2])
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.variable_num = data_shape[1]
            self.modality_num = 1
            T = self.data.shape[0]
            N = self.data.shape[1]
            M = self.data.shape[2]
            trainset = np.zeros((self.subset_num, train_end, sample_length, N))
            validset = np.zeros((self.subset_num, valid_end-train_end, sample_length, N))
            testset  = np.zeros((self.subset_num, sample_num-valid_end, sample_length, N))
            for sub in range(self.subset_num):
                for i in range(train_end):
                    sample = self.data[i:i+sample_length, :, sub]
                    trainset[sub, i] = sample
                for i in range(train_end, valid_end):
                    sample = self.data[i:i+sample_length, :, sub]
                    validset[sub, i-train_end] = sample
                for i in range(valid_end, sample_num):
                    sample = self.data[i:i+sample_length, :, sub]
                    testset[sub, i-valid_end] = sample
            self.trainset = trainset
            self.validset = validset
            self.testset  = testset

    def get_subset_num(self):
        return self.subset_num

    def get_tensor_shape(self):
        return (self.variable_num, self.modality_num)
    
    def get_data_shape(self):
        return (self.time_range, self.variable_num, self.modality_num)

    # def get_normalizer(self, norm='none'):
    #     if norm == 'none':
    #         return DoNothing()
    #     elif norm == 'sklearn':
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data = self.data[:train_range]
    #         # normalization and inverse
    #         scaler = Sklearn_StandNormalizer(train_data)
    #         return scaler
    #     elif norm == 'std':
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data = self.data[:train_range]
    #         # normalization and inverse
    #         scaler = StandNormalizer(train_data)
    #         return scaler
    #     else:
    #         raise ValueError(f'unknown normalizer: {norm}...')
        
    def init_normalizer(self, norm, train_data):
        if norm == 'none':
            return DoNothing()
        elif norm == 'sklearn':
            return Sklearn_StandNormalizer(train_data)
        elif norm == 'std':
            return StandNormalizer(train_data)
        else:
            raise ValueError(f'unknown normalizer: {norm}...')

'''
Name: Multivar Dataset Manager:
Dataset format: 
(time_series_num, time_range, dim_range)
    - time_series_num: number of time series
    - time_range: number of time steps
    - dim_range: number of dimensions
'''
class MTS_DatasetManager:
    def __init__(self, pkl_path:str, his_len:int, pred_len:int, normalizer_name:str='none',
                 test_ratio=0.1, valid_ratio=0.1, seed:int=2024, data_mode:int=0) -> None:
        random.seed(seed)
        self.his_len = his_len
        self.pred_len = pred_len
        self.normalizer_name = normalizer_name
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        # split dataset
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        
        # set data mode
        self.make_datasets(data_mode, normalizer_name)

        # random shuffle the dataset
        random.shuffle(self.trainset)
        random.shuffle(self.validset)
        random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }

    def make_datasets(self, data_mode:int, normalizer_name:str='none'):
        self.data = self.data_pkl['data']
        sample_length = self.his_len + self.pred_len
        sample_num = self.data.shape[0] - sample_length
        train_end = int(sample_num * self.train_ratio)
        valid_end = int(sample_num * self.valid_ratio) + train_end
        self.raw_train_data = self.data[:, :train_end]
        # normalize data
        self.normalizer = self.init_normalizer(normalizer_name, self.raw_train_data)
        self.data = self.normalizer.transform(self.data)
        # the shape of data read from .pkl is (T, N, M),
        # set data mode to change the shape
        # - 0: Channel-Modality-Mixing      (T, NxM, 1)       (Train one model)
        # - 1: Channel-Modality-Independent (T x N x M, 1, 1) (Train one model)
        # - 2: Modality-Independent         (T x M, N)        (Trian one model)
        # - 3: Modality-Individual          (T, N) x M        (Train M models)
        if data_mode == 0:
            # Channel-Modality-Mixing
            # (T, N, M) -> (1, T, NxM, 1)
            self.subset_num = 1
            self.time_range = int(self.data.shape[0])
            self.variable_num = int(self.data.shape[1] * self.data.shape[2])
            self.modality_num = 1
            self.data = self.data.reshape(self.time_range, -1, 1)
            # self.data = np.expand_dims(self.data, axis=-1)
            
            sample_list = []
            for i in range(sample_num):
                data = self.data[i: i+sample_length]
                # print(data.shape)
                sample_list.append(data)
            self.data = np.array(sample_list)
            # self.data = np.concatenate(sample_list, axis=0)
            self.data = np.expand_dims(self.data, axis=0)
            self.trainset = self.data[:, :train_end]
            self.validset = self.data[:, train_end:valid_end]
            self.testset  = self.data[:, valid_end:]
        elif data_mode == 1:
            # Channel-Modality-Independent
            # (T, N, M) -> (1, T x N x M, 1, 1)
            self.subset_num = 1
            self.time_range = int(self.data.shape[0])
            self.modality_num = 1
            self.variable_num = 1
            trainset = []
            validset = []
            testset = []
            T = self.data.shape[0]
            N = self.data.shape[1]
            M = self.data.shape[2]
            # trainset
            for i in range(train_end):
                for j in range(N):
                    for k in range(M):
                        sample = self.data[i:i+sample_length, j, k]
                        trainset.append(sample)
            # validset
            for i in range(train_end, valid_end):
                for j in range(N):
                    for k in range(M):
                        sample = self.data[i:i+sample_length, j, k]
                        validset.append(sample)
            # testset
            for i in range(valid_end, sample_num):
                for j in range(N):
                    for k in range(M):
                        sample = self.data[i:i+sample_length, j, k]
                        testset.append(sample)
            self.trainset = np.array(trainset)
            self.trainset = np.expand_dims(self.trainset, axis=0)
            self.trainset = np.expand_dims(self.trainset, axis=-1)
            self.validset = np.array(validset)
            self.validset = np.expand_dims(self.validset, axis=0)
            self.validset = np.expand_dims(self.validset, axis=-1)
            # self.trainset = np.expand_dims(self.validset, axis=-1)
            self.testset  = np.array(testset)
            self.testset  = np.expand_dims(self.testset, axis=0)
            self.testset = np.expand_dims(self.testset, axis=-1)
            # self.trainset = np.expand_dims(self.testset, axis=-1)

        elif data_mode == 2:
            # Modality-Independent
            # (T, N, M) -> (1, T x M, N, 1)
            self.subset_num = 1
            self.time_range = int(self.data.shape[0])
            self.modality_num = 1
            self.variable_num = int(self.data.shape[1])
            trainset = []
            validset = []
            testset = []
            T = self.data.shape[0]
            N = self.data.shape[1]
            M = self.data.shape[2]
            # trainset
            for i in range(train_end):
                for j in range(M):
                    sample = self.data[i:i+sample_length, :, j]
                    trainset.append(sample)
            # validset
            for i in range(train_end, valid_end):
                for j in range(M):
                    sample = self.data[i:i+sample_length, :, j]
                    validset.append(sample)
            # testset
            for i in range(valid_end, sample_num):
                for j in range(M):
                    sample = self.data[i:i+sample_length, :, j]
                    testset.append(sample)
            self.trainset = np.array(trainset)
            self.trainset = np.expand_dims(self.trainset, axis=0)
            self.trainset = np.expand_dims(self.trainset, axis=-1)
            self.validset = np.array(validset)
            self.validset = np.expand_dims(self.validset, axis=0)
            self.validset = np.expand_dims(self.validset, axis=-1)
            self.testset  = np.array(testset)
            self.testset  = np.expand_dims(self.testset, axis=0)
            self.testset = np.expand_dims(self.testset, axis=-1)

        elif data_mode == 3:
            # Modality-Individual
            # (T, N, M) -> (M, T, N, 1)
            self.subset_num = int(self.data.shape[2])
            self.time_range = int(self.data.shape[0])
            self.modality_num = 1
            self.variable_num = int(self.data.shape[1])
            T = self.data.shape[0]
            N = self.data.shape[1]
            M = self.data.shape[2]
            trainset = np.zeros((self.subset_num, train_end, sample_length, N))
            validset = np.zeros((self.subset_num, valid_end-train_end, sample_length, N))
            testset  = np.zeros((self.subset_num, sample_num-valid_end, sample_length, N))
            for sub in range(self.subset_num):
                for i in range(train_end):
                    sample = self.data[i:i+sample_length, :, sub]
                    trainset[sub, i] = sample
                for i in range(train_end, valid_end):
                    sample = self.data[i:i+sample_length, :, sub]
                    validset[sub, i-train_end] = sample
                for i in range(valid_end, sample_num):
                    sample = self.data[i:i+sample_length, :, sub]
                    testset[sub, i-valid_end] = sample
            self.trainset = trainset
            self.validset = validset
            self.testset  = testset

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_subset_num(self):
        return self.subset_num

    def get_modality_num(self):
        return self.modality_num
    
    def get_variable_num(self):
        return self.variable_num
    
    def get_tensor_shape(self):
        return (self.variable_num, self.modality_num)

    def get_data_shape(self):
        return (self.time_range, self.variable_num, self.modality_num)

    def get_his_pred_from_idx(self, time_series_idx:int, idx:int):
        if time_series_idx >= self.subset_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.subset_num}")
        # his = self.data[time_series_idx, idx: int(idx+self.his_len)]
        # pred = self.data[time_series_idx, int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        data = self.data[time_series_idx, idx]
        his = data[:self.his_len]
        pred = data[self.his_len:]
        return his, pred
    
    def get_seq_from_idx(self, time_series_idx:int ,idx:int):
        if time_series_idx >= self.subset_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.subset_num}")
        data = self.data[time_series_idx, idx] 
        return data

    def init_normalizer(self, norm, train_data):
        if norm == 'none':
            return DoNothing()
        elif norm == 'sklearn':
            return Sklearn_StandNormalizer(train_data)
        elif norm == 'std':
            return StandNormalizer(train_data)
        else:
            raise ValueError(f'unknown normalizer: {norm}...')

    # def get_normalizer(self, norm='none')->list:
    #     if norm == 'none':
    #         normalizer_list = []
    #         for i in range(self.subset_num):
    #             normalizer_list.append(DoNothing())
    #         return normalizer_list
    #     elif norm == 'sklearn':
    #         normalizer_list = []
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data  = self.data[:, :train_range]
    #         for i in range(self.subset_num):
    #             train_data_i = train_data[i]
    #             scaler = Sklearn_StandNormalizer(train_data_i)
    #             normalizer_list.append(scaler)
    #         return normalizer_list
    #     elif norm == 'std':
    #         normalizer_list = []
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data  = self.data[:, :train_range]
    #         for i in range(self.subset_num):
    #             train_data_i = train_data[i]
    #             scaler = StandNormalizer(train_data_i)
    #             normalizer_list.append(scaler)
    #         return normalizer_list
    #     else:
    #         raise ValueError(f'unknown normalizer: {norm}...')