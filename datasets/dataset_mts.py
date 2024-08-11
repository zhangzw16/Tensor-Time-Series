import os
import pickle as pkl

import numpy as np
from torch.utils.data import DataLoader

from .normalizer import IdentityNormalizer, StandNormalizer
from .tensor_dataset import TensorTSDataset


class MTS_Dataset:
    """
    Name: Multivar Dataset:
    Dataset format:
    (time_series_num, time_range, dim_range)
        - time_series_num: number of time series
        - time_range: number of time steps
        - dim_range: number of dimensions
    """

    def __init__(
        self,
        pkl_path: str,
        his_len: int,
        pred_len: int,
        test_ratio=0.1,
        valid_ratio=0.1,
        # seed: int = 2024,
        data_mode: int = 0,
        # phase: str = "train",  # train, valid, test
        seperate: bool = False,
    ) -> None:
        # random.seed(seed)
        self.his_len = his_len
        self.pred_len = pred_len
        self.seperate = seperate

        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, "rb") as file:
            self.data_pkl = pkl.load(file)
        # set data mode
        self.set_data_mode(data_mode)
        # split dataset
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        if train_ratio < 0:
            raise ValueError(
                f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}"
            )
        # default step = 1
        # split:
        # | <--- trian ---> | <- valid ->  | <- test -> |
        train_start_idx = 0
        train_end_idx = int(self.time_range * train_ratio)
        val_start_idx = int(self.time_range * train_ratio) - his_len
        val_end_idx = int(self.time_range * (train_ratio + valid_ratio))
        test_start_idx = int(self.time_range * (train_ratio + valid_ratio)) - his_len
        test_end_idx = int(self.time_range)
        train = self.data[:, train_start_idx:train_end_idx]
        valid = self.data[:, val_start_idx:val_end_idx]
        test = self.data[:, test_start_idx:test_end_idx]
        self.data_map = {
            "train": train,
            "valid": valid,
            "test": test,
        }

    def set_data_mode(self, data_mode: int):
        self.data_mode = data_mode
        self.data = self.data_pkl["data"]
        # raw data shape: (time, dim1, dim2)
        # set data mode to change the shape
        # - 0: (1, time, dim1*dim2, 1)
        # - 1: (dim2, time, dim1, 1)
        # - 2: (dim1, time, dim2, 1)
        if data_mode == 0:
            self.data = self.data.reshape(1, self.data.shape[0], -1, 1)
        elif data_mode == 1:
            self.data = self.data.transpose(2, 0, 1)
            self.data = np.expand_dims(self.data, axis=-1)
        elif data_mode == 2:
            self.data = self.data.transpose(1, 0, 2)
            self.data = np.expand_dims(self.data, axis=-1)
        data_shape = self.data.shape
        self.time_series_num = int(data_shape[0])
        self.time_range = int(data_shape[1])
        self.dim_range = int(data_shape[2])

    def get_time_series_num(self):
        return self.time_series_num

    def get_dim_num(self):
        return self.dim_range

    def get_data_shape(self):
        return (self.get_time_series_num(), self.time_range, self.dim_range)

    def get_dataloaders(self, phase: str, batch_size=16, drop_last=False):
        if self.data_mode == 0:
            dataset = TensorTSDataset(
                self.data_map[phase][0], self.his_len, self.pred_len
            )
            return [DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)]
        else:
            dataloaders = []
            for i in range(self.time_series_num):
                dataset = TensorTSDataset(
                    self.data_map[phase][i], self.his_len, self.pred_len
                )
                dataloaders.append(
                    DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
                )
            return dataloaders

    def get_normalizer(self, norm="none") -> list:
        if norm == "none":
            normalizer_list = []
            for i in range(self.time_series_num):
                normalizer_list.append(IdentityNormalizer())
            return normalizer_list
        elif norm == "std":
            normalizer_list = []
            train_data = self.data_map["train"]
            for i in range(self.time_series_num):
                train_data_i = train_data[i]
                scaler = StandNormalizer(
                    mean=np.mean(train_data_i), std=np.std(train_data_i)
                )
                normalizer_list.append(scaler)
            return normalizer_list
        else:
            raise ValueError(f"unknown normalizer: {norm}...")
