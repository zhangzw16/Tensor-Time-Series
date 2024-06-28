import torch.nn as nn
import torch
import yaml
import numpy as np
import os

from models.model_base import StatModelBase

class HM(nn.Module):
    def __init__(self, his_len:str, pred_len:str, dim1=1, dim2=1) -> None:
        super(HM, self).__init__()
        self.his_len = his_len
        self.pred_len = pred_len
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        # (batch, time, dim1, dim2)
        batch, time, x_d1, x_d2 = x.size()
        mean = torch.mean(x,dim=(0,1))
        pred = torch.ones((batch, self.pred_len, x_d1, x_d2)).to(x.device)
        pred[:, :] = mean
        return pred

'''
History Mean Model
'''
class HM_StatModel(StatModelBase):
    def __init__(self, configs: dict) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()
        self.init_others()

    def init_model(self, args=...) -> nn.Module:
        # task configs
        tensor_shape = self.configs['tensor_shape']
        self.input_tensor_shape = tensor_shape
        self.n_his = self.configs['his_len']
        self.n_pred = self.configs['pred_len']
        self.normalzier = self.configs['normalizer']
        # model parameters configs
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))

        self.model = HM(self.n_his, self.n_pred, self.input_tensor_shape[0], self.input_tensor_shape[1])
        
    def init_others(self, dataset_name=None):
        pass

    def forward(self, x, aux_info:dict={}):
        # x = (batch, time, dim1, dim2)
        # HW <- (batch, time, dim1, dim2)
        value = x[:, :, :self.input_tensor_shape[0], :self.input_tensor_shape[1]]
        in_data = value[:, :self.n_his]
        truth = value[:, self.n_his:self.n_his+self.n_pred]

        # normalization
        in_data = self.normalzier.transform(in_data)
        pred = self.model(in_data)
        # inverse
        pred = self.normalzier.inverse_transform(pred)

        return pred, truth
        
    def backward(self, loss):
        pass

    def get_loss(self, pred, truth):
        loss = torch.mean(pred-truth)
        return loss
    