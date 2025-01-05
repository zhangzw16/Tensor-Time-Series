import torch
import numpy as np
from .dataset import TTS_DatasetManager, MTS_DatasetManager, Stat_DatasetManager
from torch.utils.data import Dataset

class TTS_Dataset_Torch(Dataset):
    def __init__(self, dataset_manager:TTS_DatasetManager, name:str, subset_idx:int=0):
        self.dataset_manager = dataset_manager
        self.his_len = dataset_manager.his_len
        self.pred_len = dataset_manager.pred_len
        self.time_range = dataset_manager.time_range
        self.subset_idx = subset_idx
        self.dataset = dataset_manager.get_dataset(name)[subset_idx]
        # print(self.dataset.shape); exit()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # data_index = self.data_index[idx]
        # seq = self.dataset_manager.get_seq_from_idx(data_index)
        # seq = torch.from_numpy(seq).float()
        seq = self.dataset[idx]
        seq = torch.from_numpy(seq).float()
        return seq
    
class MTS_Dataset_Torch(Dataset):
    def __init__(self, dataset_manager:MTS_DatasetManager, name:str, subset_idx:int=0):
        self.dataset_manager = dataset_manager
        self.his_len = dataset_manager.his_len
        self.pred_len = dataset_manager.pred_len
        self.time_range = dataset_manager.time_range
        self.subset_idx = subset_idx
        self.dataset = dataset_manager.get_dataset(name)[subset_idx]
        # print(name, self.dataset.shape); exit()
        if len(self.dataset.shape) == 3:
            self.dataset = np.expand_dims(self.dataset, axis=3)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # data_index = self.dataset[idx]
        # seq = self.dataset_manager.get_seq_from_idx(self.ts_idx, data_index)
        # seq = torch.from_numpy(seq).float()
        seq = self.dataset[idx]
        seq = torch.from_numpy(seq).float()
        return seq
    
class Stat_Dataset_Torch(Dataset):
    def __init__(self, dataset_manager:Stat_DatasetManager, name:str, subset_idx:int=0):
        self.dataset_manager = dataset_manager
        self.his_len = dataset_manager.his_len
        self.pred_len = dataset_manager.pred_len
        self.time_range = dataset_manager.time_range
        self.subset_idx = subset_idx
        self.dataset = dataset_manager.get_dataset(name)[subset_idx]
        # print(self.dataset.shape); exit()
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        seq = self.dataset[idx]
        seq = torch.from_numpy(seq).float()
        return seq
    

