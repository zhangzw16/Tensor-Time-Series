import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
from models.model_base import MultiVarModelBase

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]

class NLTSF_MultiVarModel(MultiVarModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_others()
        self.init_model()

    def init_others(self):
        # get data shape
        self.data_shape = self.configs["tensor_shape"]
        self.channels = self.data_shape[0]
    
    def init_model(self, args=...):
        # load task parameters
        self.input_len = self.configs["his_len"]
        self.pred_len = self.configs["pred_len"]
        self.device = self.configs["task_device"]
        self.normalizer = self.configs["normalizer"]

        # load model params
        model_configs_yaml = os.path.join(os.path.dirname(__file__), "model.yml")
        model_configs = yaml.safe_load(open(model_configs_yaml))
        self.loss_name = model_configs["loss"]
        self.lr = model_configs["lr"]
        self.weight_decay = model_configs["weight_decay"]

        # configure loss function
        if self.loss_name == "mse":
            self.loss = nn.MSELoss()
        elif self.loss_name == "mae":
            self.loss = nn.L1Loss()
        else:  # mse loss by default
            self.loss = nn.MSELoss()
        
        # build model
        self.model = nn.ModuleList()
        self.unit = nn.Linear(self.input_len, self.pred_len)
        for i in range(0, self.channels):
            self.model.append(self.unit)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            eps=1.0e-8,
            weight_decay=self.weight_decay,
            amsgrad=False)
        
    def set_device(self, device):
        self.model.to(device)
    
    def forward(self, x, aux_info=...):
        # input shape: (batch, time(hist+pred), dim1*dim2, 1)
        # model need: (batch, time(hist), dim1*dim2), so squeeze needed
        x1 = x[:, : self.input_len, :, :]
        x_hist = x1.squeeze(-1)
        x_hist = self.normalizer.transform(x_hist)  # normalizer added
        # pre-processing
        seq_last = x_hist[:, -1, :].detach()
        seq_last = seq_last.unsqueeze(1)
        x_in = x_hist - seq_last
        # forward
        outputs = torch.zeros([x_in.size(0), self.pred_len, x_in.size(2)], dtype=x_in.dtype).to(self.device)
        for i in range(0, self.channels):
            outputs[:, :, i] = self.model[i](x_in[:, :, i])
        outputs = outputs + seq_last
        # return
        outputs = self.normalizer.inverse_transform(outputs)
        y_pred = outputs.unsqueeze(-1)
        truth = x[:, self.input_len:self.input_len+self.pred_len, :, :]

        return y_pred, truth
    
    def backward(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optim.step()

    def get_loss(self, pred, truth):
        L = self.loss(pred, truth)
        return L