import torch.nn as nn
import torch
import yaml
import numpy as np
import pickle
import os

from models.model_base import TensorModelBase
from .networks.tgcn import TGCN
from .networks.tlstm import TLSTM
from .networks.output import Output
from .networks import utils

'''
NET3 model
'''
class NET3(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        if configs == None:
            pre_dir = os.path.dirname(__file__)
            configs = yaml.safe_load(open(os.path.join(pre_dir, 'model.yml')))
        self.configs = self.default_configs()
        utils.update_configs(configs, self.configs)

        self.tgcn = TGCN(self.configs["TGCN"])
        self.configs["TLSTM"]["dim_input"] = self.tgcn.configs["dim_output"]
        utils.update_configs(self.tgcn.configs, self.configs["TGCN"])

        self.configs["TLSTM"]["mode_dims"] = self.configs["mode_dims"]
        self.tlstm = TLSTM(self.configs["TLSTM"])
        utils.update_configs(self.tlstm.configs, self.configs["TLSTM"])

        self.configs["Output"]["mode_dims"] = self.configs["mode_dims"]
        self.configs["Output"]["dim_input"] = self.tgcn.configs["dim_output"] + self.tlstm.configs["dim_output"]
        self.output = Output(self.configs["Output"])

    def forward(self, values, adj, hx=None, indicators=None):
        """
        :param values: [batch_size, n_1, n_2, ..., n_M, t_step] values of tensors
        :param adj: adjacency matrix dictionary {n_m: A_m}
        :param hx: (h, c) [batch_size, n_1, n_2, ..., n_M, dim_emb] for both h and c
        :param indicators: [batch_size, n_1, n_2, ..., n_M, t_step] 0: missing value, 1: valid value
        """
        if indicators is None:
            indicators = torch.ones_like(values, dtype=torch.float)

        n_steps = values.shape[-1]
        out_list = []
        for t in range(n_steps):
            if t == 0:
                emb = (values[..., t]*indicators[..., t]).unsqueeze(-1)   # [batch_size, n_1, n_2, ..., n_M, dim]
            else:
                v = indicators[..., t]*values[..., t] + (1 - indicators[..., t])*out_list[-1]  # fill the missing values
                emb = (v * torch.ones_like(v)).unsqueeze(-1)

            emb_gcn = self.tgcn(inputs=emb, adj=adj)
            h_t, hx = self.tlstm(emb_gcn, hx)
            h_t = torch.cat([h_t, emb_gcn], dim=-1)
            out_t = self.output(h_t).squeeze()
            out_list.append(out_t)
        output = torch.stack(out_list, dim=-1)
        return output, hx

    @classmethod
    def default_configs(cls):
        return {
            "mode_dims": {0: 54, 1: 4},    # required for building the model
            "TGCN": TGCN.default_configs(),
            "TLSTM": TLSTM.default_configs(),
            "Output": Output.default_configs(),
        }


'''
Tensor Model: NET3
'''
class NET3_TensorModel(TensorModelBase):
    def __init__(self, configs: dict={}) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()
        self.init_others()

    def init_model(self, args={}) -> NET3:
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        self.orthogonal_weight = 1e-3
        self.reconstruction_weight = 1e-3
        tensor_shape = self.configs['tensor_shape']
        self.tensor_shape = tensor_shape
        model_configs['mode_dims'] = {0: tensor_shape[0], 1:tensor_shape[1]}       
        self.normalizer = self.configs['normalizer']
        self.model = NET3(model_configs) 
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = utils.mse_loss

    def init_others(self, dataset_name=None):
        # network_dir = "/home/zhuangjiaxin/workspace/Tensor-Time-Series/repos/NET3/dataset_processed/future/networks.pkl"
        # networks = pickle.load(
        #     open(os.path.join(network_dir), 'rb')
        # )
        # for n in networks:
        #     networks[n] = torch.from_numpy(networks[n]).float()
        self.network = {}
        for i in range(len(self.tensor_shape)):
            dim = self.tensor_shape[i]
            adj_matrix = np.random.rand(dim, dim)
            adj_matrix = torch.from_numpy(adj_matrix).float()
            self.network[i] = adj_matrix
    
    def set_device(self, device='cpu'):
        self.model.to(device)
        for n in self.network:
            self.network[n] = self.network[n].to(device)

    def forward(self, input, aux_info:dict={}):
        # the shape pf input tensor: (batch, time, dim1, dim2)
        # but the shape pf input tensor in NET3: (batch, dim1, dim2, time)
        # therefore, permute the input tensor
        value = input.permute(0,2,3,1)
        dim1 = self.model.configs['mode_dims'][0]
        dim2 = self.model.configs['mode_dims'][1]
        value = value[:, :dim1, :dim2, :]   # ensure the correct shape (test)
        value = self.normalizer.transform(value)
        adj = self.network
        pred, hx = self.model(values=value[...,:-1], adj=adj)        
        model_pred = pred[..., -1]
        model_pred = self.normalizer.inverse_transform(model_pred)
        truth = value[..., -1]
        return model_pred, truth
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        

    def get_loss(self, pred, truth):
        indicators = torch.ones_like(pred, dtype=torch.float)
        if self.model.configs["TLSTM"]['is_decompose']:
            return self.get_loss_rmse(truth, pred, indicators) + \
                   self.orthogonal_weight * self.get_loss_orthogonal() + \
                   self.reconstruction_weight * self.get_loss_reconstruction()
        return self.get_loss_rmse(truth, pred, indicators)
    def get_loss_rmse(self, y, y_pred, indicators):
        return torch.sqrt(self.criterion(y=y, y_pred=y_pred, indicators=indicators))
    def get_loss_orthogonal(self):
        return self.model.tlstm.get_orthogonal_loss()
    def get_loss_reconstruction(self):
        return self.model.tlstm.get_reconstruction_loss()
    