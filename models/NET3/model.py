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
        # print(values.shape, indicators.shape);exit()
        n_steps = values.shape[-1]
        out_list = []
        for t in range(n_steps):
            if t == 0:
                emb = (values[..., t]*indicators[..., t]).unsqueeze(-1)   # [batch_size, n_1, n_2, ..., n_M, dim]
            else:
                
                out_last = out_list[-1]
                # print(out_last.shape)
                # if out_last.size(-1) != 2:
                #     # out_last = out_last.permute(2,0,1)
                #     out_last = out_last.squeeze(-1)
                #     out_last = out_last.unsqueeze(0)
                # print(indicators.shape, values.shape, out_last.shape);exit()
                # print(out_last.shape, indicators[...,t].shape);exit()
                dim1, dim2, dim3 = indicators[..., t].size()
                out_last = out_last.view((dim1,dim2,dim3))
                v = indicators[..., t]*values[..., t] + (1 - indicators[..., t])*out_last  # fill the missing values
                emb = (v * torch.ones_like(v)).unsqueeze(-1)

            emb_gcn = self.tgcn(inputs=emb, adj=adj)
            h_t, hx = self.tlstm(emb_gcn, hx)
            h_t = torch.cat([h_t, emb_gcn], dim=-1)
            out_t = self.output(h_t).squeeze()
            if len(out_t.shape) == 2:
                out_t = out_t.unsqueeze(-1)
            out_list.append(out_t)
            # print(emb.shape, emb_gcn.shape, h_t.shape, out_t.shape)
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
        self.graph_generator = self.configs['graphGenerator']
        self.model = NET3(model_configs) 
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = utils.mse_loss

    def init_others(self, dataset_name=None):
        self.network = {}
        # self.network[0] = torch.from_numpy(self.graph_generator.cosine_similarity_matrix(n_dim=0, normal=True)).float()
        # self.network[1] = torch.from_numpy(self.graph_generator.pearson_matrix(n_dim=1, normal=True)).float()
        self.network[0] = torch.from_numpy(self.graph_generator.gen_graph(n_dim=0, normal=True)).float()
        self.network[1] = torch.from_numpy(self.graph_generator.gen_graph(n_dim=1, normal=True)).float()

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
        in_value = self.normalizer.transform(value[...,:-1])
        adj = self.network
        pred, hx = self.model(values=in_value, adj=adj)        
        # print(f"out: pred:{pred.shape}, truth: {value.shape}");exit()
        if value.shape[0] != pred.shape[0]:
            d1, d2, d3, d4 = value.size()
            pred = pred.view((d1,d2,d3,d4-1))
        model_pred = self.normalizer.inverse_transform(pred)
        model_pred = pred[..., -1]
        truth = value[..., -1]
        # print(model_pred.shape, truth.shape);exit()
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