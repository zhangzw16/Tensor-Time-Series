import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import math
from .layers.dcrnn_model import DCRNNModel
from models.model_base import TensorModelBase   

class DCRNN_TensorModel(TensorModelBase):
    def __init__(self, configs: dict = ...) -> None:
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
        # self.batch_size = self.configs['batch_size']
        # read model configs
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        
        self.enc_input_dim = self.tensor_shape[1]
        self.dec_input_dim = self.tensor_shape[1]

        self.max_diffusion_step = model_configs['max_diffusion_step']
        self.num_rnn_layers = model_configs['num_rnn_layers']
        self.rnn_units = model_configs['rnn_units']
        self.filter_type = model_configs['filter_type']

        # adj
        self.graph_generator = self.configs['graphGenerator']
        self.adj_mat = self.graph_generator.gen_graph(n_dim=0, normal=True)

        self.model = DCRNNModel(self.adj_mat, self.enc_input_dim, self.dec_input_dim,
                                self.max_diffusion_step, self.tensor_shape[0],self.num_rnn_layers,
                                 self.rnn_units, self.input_len, self.pred_len, self.tensor_shape[1], self.filter_type)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01, eps=1.0e-3, amsgrad=True, weight_decay=0)
        self.criterion = nn.MSELoss()

    
    def init_others(self, dataset_name=None):
        self.global_step = 0
        self.cl_decay_steps = 2000
    
    def forward(self, x, aux_info:dict={}):
        # x [batch, time, dim1, dim2]
        # DCRNN [batch, time, dim1, dim2]
        value = x[:, :, :self.tensor_shape[0], :self.tensor_shape[1]]
        source_data = value[:, :self.input_len, :, :]
        truth = value[:, self.input_len:self.input_len+self.pred_len, :, :]
        if self.model.training:
            target = truth
        else:
            # target = torch.rand_like(truth)
            target = torch.zeros_like(truth)
        # normalize
        source_data = self.normalizer.transform(source_data)
        teacher_forcing_ratio = self._compute_sampling_threshold(self.global_step, self.cl_decay_steps)
        pred = self.model(source_data, target, teacher_forcing_ratio)
        self.global_step += 1
        # inverse transform
        pred = self.normalizer.inverse_transform(pred)
        pred = pred.permute(1,0,2,3)
        # print(pred.shape, truth.shape);exit()
        return pred, truth

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def get_loss(self, pred, truth):
        loss = self.criterion(pred, truth)
        return loss
    
