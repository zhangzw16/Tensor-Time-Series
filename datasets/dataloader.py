import torch
import numpy as np
from .dataset import TTS_Dataset, MTS_Dataset

class TTS_DataLoader:
    def __init__(self, dataset:TTS_Dataset, name:str,
                 batch_size=16, drop_last=False) -> None:
        # load data
        self.dataset = dataset
        self.his_len = dataset.his_len
        self.pred_len = dataset.pred_len
        self.time_range = dataset.time_range
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_index = self.dataset.get_dataset(name)
        # compute batches
        window_size = self.his_len + self.pred_len
        batch_sample = int(window_size*batch_size)
        self.num_batch = (self.time_range//batch_sample)
        if (self.time_range % batch_sample != 0) and not drop_last:
            self.num_batch += 1
        self.batch_index = 0

    def get_batch(self, separate=False):
        auxiliary_info = {}
        for i in range(self.num_batch):
            begin = i * self.batch_size
            end = begin + self.batch_size
            batch_data_index = self.data_index[begin: end]
            auxiliary_info['idxs'] = np.array(batch_data_index)
            if separate==False:    
                # separate his and pred, if separate == True, return a whole seq
                seq_list = []
                for idx in batch_data_index:
                    seq = self.dataset.get_seq_from_idx(idx)
                    seq_list.append(seq)
                seq = np.stack(seq_list)
                seq = torch.from_numpy(seq).float()
                yield seq, auxiliary_info
            else:   
                # if separate == False, return [his, pred]
                his_list = []
                pred_list = []
                for idx in batch_data_index:
                    his, pred = self.dataset.get_his_pred_from_idx(idx)
                    his_list.append(his)
                    pred_list.append(pred)
                his = np.vstack(his_list)
                pred = np.vstack(pred_list)
                his = torch.from_numpy(his).float()
                pred = torch.from_numpy(pred).float()
                # his: (batch_size, time, dim1, dim2)
                # pred: (batch_szie, time, dim1, dim2)
                yield his, pred, auxiliary_info

class MTS_DataLoader:
    def __init__(self, dataset:MTS_Dataset, name:str,
                 batch_size=16, drop_last=False) -> None:
        # load data
        # (num, time, dim, 1)
        self.dataset = dataset
        self.his_len = dataset.his_len
        self.pred_len = dataset.pred_len
        self.time_range = dataset.time_range
        self.time_series_num = dataset.time_series_num
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_index = self.dataset.get_dataset(name)
        # compute batches
        window_size = self.his_len + self.pred_len 
        batch_sample_size = int(window_size*batch_size)
        self.num_batch = (self.time_range//batch_sample_size)
        if (self.time_range % batch_sample_size != 0) and not drop_last:
            self.num_batch += 1
        self.batch_index = [0 for _ in range(self.time_series_num)]

    def get_batch(self, time_series_idx:int, separate=False):
        auxiliary_info = {}
        for i in range(self.num_batch):
            begin = i * self.batch_size
            end = begin + self.batch_size
            batch_data_index = self.data_index[begin:end]
            auxiliary_info['idxs'] = np.array(batch_data_index)
            if separate == False:
                seq_list = []
                for idx in batch_data_index:
                    seq = self.dataset.get_seq_from_idx(time_series_idx, idx)
                    seq_list.append(seq)
                seq = np.stack(seq_list)
                seq = torch.from_numpy(seq).float()
                yield seq, auxiliary_info
            else:
                his_list = []
                pred_list = []
                for idx in batch_data_index:
                    his, pred = self.dataset.get_his_pred_from_idx(time_series_idx, idx)
                    his_list.append(his)
                    pred_list.append(pred)
                his = np.stack(his_list)
                pred = np.stack(pred_list)
                his = torch.from_numpy(his).float()
                pred = torch.from_numpy(pred).float()
                yield his, pred, auxiliary_info