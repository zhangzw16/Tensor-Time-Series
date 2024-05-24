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
    def __init__(self, configs:dict={}) -> None:
        self.configs = configs
        self.ModelType = 'BaseModel'
    def init_model(self, args={}):
        pass        
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
    def save_model(self, path:str):
        torch.save(self.model.state_dict(), path)
    def load_model(self, path:str):
        self.model.load_state_dict(torch.load(path))

'''
Tensor based model:
'''
class TensorModelBase(ModelBase):
    def __init__(self, configs:dict={}) -> None:
        super().__init__(configs)
        self.ModelType = 'Tensor'
    def init_model(self, args={}):
        pass
    def init_others(self, dataset_name=None):
        pass
    def forward(self, x, aux_info:dict={}):
        pass
    def backward(self, loss):
        pass
    def get_loss(self, pred, truth):
        pass

'''
Multivariate based model:
'''
class MultiVarModelBase(ModelBase):
    def __init__(self, configs:dict={}) -> None:
        super().__init__(configs)
        self.ModelType = 'MultiVar'
        
    def init_model(self, args={}):
        pass

    def forward(self, x, aux_info:dict={}):
        pass

    def backward(self, loss):
        pass

    def get_loss(self, pred, truth):
        pass

'''
Statistic based model:
'''
class StatModelBase(ModelBase):
    def __init__(self, configs:dict={}) -> None:
        super().__init__(configs)
        self.ModelType = 'Statistic'
        
    def init_model(self, args={}):
        pass

    def forward(self, x, aux_info:dict={}):
        pass

    def backward(self, loss):
        pass

    def get_loss(self, pred, truth):
        pass