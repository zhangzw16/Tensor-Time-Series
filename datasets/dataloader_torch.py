import torch
from .dataset import TTS_DatasetManager, MTS_DatasetManager
from torch.utils.data import Dataset

class TTS_Dataset_Torch(Dataset):
    def __init__(self, dataset_manager:TTS_DatasetManager, name:str):
        self.dataset_manager = dataset_manager
        self.his_len = dataset_manager.his_len
        self.pred_len = dataset_manager.pred_len
        self.time_range = dataset_manager.time_range
        self.data_index = dataset_manager.get_dataset(name)

    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        data_index = self.data_index[idx]
        seq = self.dataset_manager.get_seq_from_idx(data_index)
        seq = torch.from_numpy(seq).float()
        return seq
    
class MTS_Dataset_Torch(Dataset):
    def __init__(self, dataset_manager:MTS_DatasetManager, name:str, ts_idx:int=0):
        self.dataset_manager = dataset_manager
        self.his_len = dataset_manager.his_len
        self.pred_len = dataset_manager.pred_len
        self.time_range = dataset_manager.time_range
        self.data_index = dataset_manager.get_dataset(name)
        self.ts_idx = ts_idx

    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        data_index = self.data_index[idx]
        seq = self.dataset_manager.get_seq_from_idx(self.ts_idx, data_index)
        seq = torch.from_numpy(seq).float()
        return seq
    

