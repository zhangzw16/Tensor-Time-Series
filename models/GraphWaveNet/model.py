import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import math
from .layers.GraphWaveNet import gwnet
from models.model_base import MultiVarModelBase, TensorModelBase

class GraphWaveNet_TensorModel(TensorModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()

    def init_model(self, args=...):
        # task configs
        self.tensor_shape = self.configs['tensor_shape']
        # self.num_vars = int(tensor_shape[0]*tensor_shape[1])
        self.his_len = self.configs['his_len']
        self.pred_len = self.configs['pred_len']
        self.normalizer = self.configs['normalizer']

        # read model configs
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        
        
        self.layers = model_configs['layers']
        self.blocks = model_configs['blocks']
        self.kernel_size = model_configs['kernel_size']
        self.residual_channels = model_configs['residual_channels']
        self.skip_channels = model_configs['skip_channels']
        self.end_channels = model_configs['end_channels']
        self.dropout = model_configs['dropout']

        self.gcn_bool = model_configs['gcn_bool']
        self.addaptadj = model_configs['addaptadj']
        self.aptinit = model_configs['aptinit']
        # graph
        if self.aptinit:
            self.supports = None
        else:
            graph = self.configs['graphGenerator'].gen_graph(n_dim=0, normal=True)
            graph = torch.from_numpy(graph).float()
            self.supports = [graph]
            self.aptinit = graph
        # model
        self.model = gwnet(self.tensor_shape[0], self.dropout, self.supports, self.gcn_bool, self.addaptadj, self.aptinit, self.tensor_shape[1],
                           self.tensor_shape[1], self.pred_len, self.residual_channels, self.residual_channels, self.skip_channels, 
                           self.end_channels, self.kernel_size, self.blocks, self.layers)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.002, weight_decay=0.0001)
        self.criterion = nn.MSELoss()
    
    def forward(self, x, aux_info: dict = ...):
        # x [batch, time, dim1, dim2]
        # gwnet [batch, dim2, dim1, time]
        value = x[:, :, :self.tensor_shape[0], :self.tensor_shape[1]]
        value = value.permute(0, 3, 2, 1)
        in_data = value[:, :, :, :self.his_len]
        truth = value[:, :, :, self.his_len:self.his_len+self.pred_len]
        # normalization
        in_data = self.normalizer.transform(in_data)
        pred = self.model(in_data)
        # inverse
        pred = self.normalizer.inverse_transform(pred)
        
        return pred, truth
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def get_loss(self, pred, truth):
        loss = self.criterion(pred, truth)
        return loss
    
    def set_device(self, device='cpu'):
        self.model.to(device)
        for i in range(len(self.supports)):
            self.supports[i] = self.supports[i].to(device)
    

    