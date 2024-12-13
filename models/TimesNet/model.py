import os
import yaml
import torch
import torch.nn as nn
import numpy as np

from .Layers.TimesNet import Model
from .Layers.losses import mape_loss,  mase_loss, smape_loss
from models.model_base import MultiVarModelBase

class TimesNet_MultiVarModel(MultiVarModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()

    def init_model(self, args=...):
        # task configs
        self.tensor_shape = self.configs['tensor_shape']
        # print(self.tensor_shape);exit()
        self.num_var = int(self.tensor_shape[0]*self.tensor_shape[1])
        self.input_len = self.configs['his_len']
        self.pred_len = self.configs['pred_len']
        # read model configs
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        self.enc_in = model_configs['enc_in']
        self.dec_in = model_configs['dec_in']
        self.c_out = model_configs['c_out']
        self.e_layers = model_configs['e_layers']
        self.d_layers = model_configs['d_layers']
        self.d_model = model_configs['d_model']
        self.d_ff = model_configs['d_ff']
        self.factor = model_configs['factor']
        self.top_k = model_configs['top_k']
        self.dropout = model_configs['dropout']
        self.freq = model_configs['freq']
        self.embed = model_configs['embed']
        self.num_kernels = model_configs['num_kernels']
        self.criterion_name = model_configs['criterion']
        self.criterion_name = 'MSE'
        
        self.model = Model(self.input_len, self.pred_len, self.num_var, self.num_var, self.num_var,
                           self.e_layers, self.d_model, self.embed, self.top_k, 
                           self.d_ff, self.num_kernels, self.freq, self.dropout)
        # update optimizer configs 
        lr = 0.001 if self.optimizer_configs['lr'] == None else self.optimizer_configs['lr']
        eps = 1.0e-8 if self.optimizer_configs['eps'] == None else self.optimizer_configs['eps']
        weight_decay = 0 if self.optimizer_configs['weight_decay'] == None else self.optimizer_configs['weight_decay']
        self.optim = torch.optim.Adam(self.model.parameters(),lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=False)
        
        # self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = self.select_criterion(self.criterion_name)

    def select_criterion(self, criterion_name='MSE'):
        if criterion_name == 'MSE':
            return nn.MSELoss()
        elif criterion_name == 'MAPE':
            return mape_loss()
        elif criterion_name == 'MASE':
            return mase_loss()
        elif criterion_name == 'SMAPE':
            return smape_loss()
    
    def forward(self, x, aux_info: dict = ...):
        # x [batch, time, dim1, dim2]
        # TimesNet [batch, time, dim1*dim2]
        # print(x.size())
        batch, time, dim1, dim2 = x.size()
        value = x.view(batch, time, dim1*dim2)
        in_data = value[:, :self.input_len, :]
        truth = value[:, self.input_len:self.input_len+self.pred_len, :]
        # normalization
        # in_data = self.normalizer.transform(in_data)
        pred = self.model(in_data, None, None, None)
        # inverse
        # pred = self.normalizer.inverse_transform(pred)

        return pred, truth
        
    def backward(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optim.step()
    
    def get_loss(self, pred, truth):
        loss = self.criterion(pred, truth)
        return loss
    
