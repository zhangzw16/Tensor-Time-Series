import torch
import numpy as np
from .dataset import TTS_Dataset

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
        for i in range(self.num_batch):
            begin = i * self.batch_size
            end = begin + self.batch_size
            batch_data_index = self.data_index[begin: end]
            if separate==False:    
                # separate his and pred, if separate == True, return a whole seq
                seq_list = []
                for idx in batch_data_index:
                    seq = self.dataset.get_seq_from_idx(idx)
                    seq_list.append(seq)
                seq = np.stack(seq_list)
                seq = torch.from_numpy(seq).float()
                yield seq
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
                # pred: (batch_szie, time, dim1, dim2) TODO: tensor or value?
                yield his, pred