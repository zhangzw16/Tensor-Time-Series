import os
import torch
import torch.nn as nn
import importlib

'''
Basic Model API
    - forward
    - get_loss
'''
class ModelBase(object):
    def __init__(self, configs:dict) -> None:
        # self.model = self.init_model()
        # self.configs = configs
        pass
    def init_model(self, args={})->nn.Module:
        return nn.Module()        
    def forward(self,x ,y):
        pass
    def backward(self, loss):
        pass
    def get_loss(self):
        pass
    # change mode
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    def set_device(self, device='cpu'):
        self.model.to(device)
    def load_param(self, path):
        pass

'''
GNN based model:
    - adjacency matrix
    - forward
    - get_loss
'''
class TensorModelBase(ModelBase):
    def __init__(self, configs: dict) -> None:
        super().__init__(configs)
        # self.network = self.init_adj_matrix()

    def init_model(self, args={})->nn.Module:
        return nn.Module()

    def init_others(self, dataset_name=None):
        pass

    def forward(self, x):
        pass

    def backward(self, loss):
        pass

    def get_loss(self, pred, truth):
        pass