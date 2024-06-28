import os
import yaml
import torch
import torch.nn as nn
import numpy as np

from .layers.net import gtnet
from .layers import utils
from models.model_base import TensorModelBase

class MTGNN_TensorModel(TensorModelBase):
    def __init__(self, configs:dict={})->None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()
        self.init_others()
    
    def init_model(self, args=...):
        # task configs
        self.tensor_shape = self.configs['tensor_shape']
        self.normalizer = self.configs['normalizer']
        self.input_len = self.configs['his_len']
        self.pred_len = self.configs['pred_len']
        self.device = self.configs['task_device']
        # read model configs
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))

        self.gcn_enable = model_configs['gcn_enable']
        self.buildA_enable = model_configs['buildA_enable']
        self.dropout = model_configs['dropout']
        self.gcn_depth = model_configs['gcn_depth']
        self.layers = model_configs['layers']
        self.subgraph_size = model_configs['subgraph_size']
        self.dilation_exponential = model_configs['dilation_exponential']
        self.predefined_A = None
        # self.predefined_A = model_configs['pedefined_A']
        # if self.predefined_A != None:
        #     pass
        
        self.conv_channels = model_configs['conv_channels']
        self.residual_channels = model_configs['residual_channels']
        self.skip_channels = model_configs['skip_channels']
        self.end_channels = model_configs['end_channels']

        self.node_dim = model_configs['node_dim']
        self.out_dim = model_configs['out_dim']
        self.idx_update_interval = model_configs['idx_update_interval']
        self.idx_update_cnt = 0
        if self.out_dim == 0:
            self.out_dim = self.tensor_shape[1]

        self.prop_alpha = model_configs['prop_alpha']
        self.tanh_alpha = model_configs['tanh_alpha']
        self.layer_norm_affline = model_configs['layer_norm_affline']

        self.model = gtnet(self.gcn_enable, self.buildA_enable, self.gcn_depth, self.tensor_shape[0],
                           self.device, predefined_A=self.predefined_A, dropout=self.dropout, subgraph_size=self.subgraph_size,
                           node_dim=self.node_dim, dilation_exponential=self.dilation_exponential, conv_channels=self.conv_channels,
                           residual_channels=self.residual_channels, skip_channels=self.skip_channels, end_channels=self.end_channels,
                           input_len=self.input_len, in_dim=self.tensor_shape[1], out_dim=self.out_dim, pred_len=self.pred_len, layers=self.layers, propalpha=self.prop_alpha, tanhalpha=self.tanh_alpha,
                           layer_norm_affline=self.layer_norm_affline)
        self.criterion = utils.masked_mae
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.clip = 5
    
    def init_others(self, dataset_name=None):
        pass
    
    def forward(self, x, aux_info: dict = ...):
        # x [batch, time, dim1, dim2]
        # MTGNN [batch, dim2, dim1, time]
        value = x[:, :, :self.tensor_shape[0], :self.tensor_shape[1]]
        value = value.permute(0,3,2,1)
        in_data = value[:, :, :, :self.input_len]
        truth = value[:, :self.out_dim, :, self.input_len:self.input_len+self.pred_len]
        # idx generation
        idx = None
        if self.model.training:
            if self.idx_update_cnt % self.idx_update_interval == 0:
                self.idx = np.random.permutation(range(self.tensor_shape[0]))
                self.idx_update_cnt = 0
            self.idx_update_cnt += 1
            idx = torch.tensor(self.idx).to(x.device)
            in_data = in_data[:, :, idx, :]
            truth = truth[:, :, idx, :]
        # normalization
        in_data = self.normalizer.transform(in_data)
        pred = self.model(in_data, idx=idx)
        pred = pred.permute(0,3,2,1)
        # inverse
        pred = self.normalizer.inverse_transform(pred)
        return pred, truth
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optim.step()

    
    def get_loss(self, pred, truth):
        loss = self.criterion(pred, truth, 0.0)
        return loss

    
