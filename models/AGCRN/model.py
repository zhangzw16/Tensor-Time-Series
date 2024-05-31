import os
import yaml
import torch
import torch.nn as nn
import numpy as np

from .Layers.AGCRN import AGCRN
from .Layers.utils import masked_mae_loss
from models.model_base import TensorModelBase

class AGCRN_TensorModel(TensorModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()
        self.init_others()

    def init_model(self, args=...):
        # task configs
        self.tensor_shape = self.configs['tensor_shape']
        # print('tensor_shape:', self.tensor_shape)
        self.normalizer = self.configs['normalizer']
        self.input_len = self.configs['his_len']
        self.pred_len = self.configs['pred_len']
        # read model configs
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        self.input_dim = self.tensor_shape[1]
        self.output_dim = self.input_dim
        # self.embed_dim = model_configs['embed_dim']
        self.embed_dim = self.tensor_shape[1]
        self.rnn_units = model_configs['rnn_units']
        self.hidden_dim = self.rnn_units
        self.num_layers = model_configs['num_layers']
        self.cheb_order = model_configs['cheb_order']
        self.default_graph = model_configs['default_graph']
        self.loss_func_name = model_configs['loss_func']
        self.grad_norm = model_configs['grad_norm']
        self.max_grad_norm = model_configs['max_grad_norm']

        self.model = AGCRN(self.tensor_shape[0], self.input_dim, self.hidden_dim, self.output_dim,
                           self.pred_len, self.num_layers, self.default_graph, self.embed_dim, self.cheb_order)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
                
        self.optim = torch.optim.Adam(self.model.parameters(),lr=0.003, eps=1.0e-8,
                                      weight_decay=0, amsgrad=False)

    def init_others(self, dataset_name=None):
        if self.loss_func_name == 'mask_mae':
            self.criterion = masked_mae_loss(self.normalizer, mask_value=0.0)
        elif self.loss_func_name == 'mae':
            self.criterion = torch.nn.L1Loss()
        elif self.loss_func_name == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f'unknown loss func: {self.loss_func_name}')
    
    def forward(self, x, aux_info: dict = ...):
        # x [batch, time, dim1, dim2]
        # AGCRN: [batch, time, dim1, dim2]
        value = x[:, :, :self.tensor_shape[0], :self.tensor_shape[1]]
        in_data = value[:, :self.input_len, :, :]
        truth = value[:, self.input_len:self.input_len+self.pred_len, :, :]
        # use label as input in the decoder for all steps (teaching_forcing is false)
        teacher_forcing_ratio = 0
        in_data = self.normalizer.transform(in_data)
        pred = self.model(in_data, truth, teacher_forcing_ratio=teacher_forcing_ratio)
        # invserse
        pred = self.normalizer.inverse_transform(pred)
        return pred, truth
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        if self.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()

    def get_loss(self, pred, truth):
        loss = self.criterion(pred, truth)
        return loss