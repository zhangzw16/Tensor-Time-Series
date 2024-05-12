import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

import os
import yaml
from .layers.stemgnn_arch import StemGNN
from models.model_base import MultiVarModelBase 

class StemGNN_MultiVarModel(MultiVarModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()
    
    def init_model(self, args=...):
        tensor_shape = self.configs['tensor_shape']
        self.num_vars = int(tensor_shape[0]*tensor_shape[1])
        self.input_len = self.configs['his_len']
        self.pred_len = self.configs['pred_len']
        self.normalizer = self.configs['normalizer']

        # model parameters config
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))

        self.stack_cnt = model_configs['stack_cnt']
        self.multi_layer = model_configs['multi_layer']
        self.dropout = model_configs['dropout']
        self.leaky_rate = model_configs['leaky_rate']

        self.model = StemGNN(self.num_vars, self.stack_cnt, 
                             self.input_len, self.multi_layer, self.pred_len, 
                             self.dropout, self.leaky_rate)

        self.criterion = nn.MSELoss(reduction='mean')
        if model_configs['optimizer'] == 'RMSProp':
            self.optim = torch.optim.RMSprop(self.model.parameters(), lr=1e-4, eps=1e-8)
        else:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
    
    def forward(self, x, aux_info: dict = ...):
        # x: (batch, time, dim1, dim2)
        # STemGNN: (batch, time, N)
        batch, time, dim1, dim2 = x.size()
        value = x.view(batch, time, dim1*dim2)
        in_data = value[:, :self.input_len, :]
        truth = value[:, self.input_len:self.input_len+self.pred_len, :]
        # normalization
        in_data = self.normalizer.transform(in_data)
        pred, _ = self.model(in_data)
        # inverse
        pred = self.normalizer.inverse_transform(pred)

        return pred, truth
    
    def backward(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optim.step()
    
    def get_loss(self, pred, truth):
        loss = self.criterion(pred, truth)
        return loss
    
    