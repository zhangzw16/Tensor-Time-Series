# reference: Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting
# paper link: https://arxiv.org/pdf/2103.16349
import torch
import torch.nn as nn
import os, yaml
from models.model_base import StatModelBase

class HistoricalInertia(nn.Module):
    def __init__(self, in_steps=12, out_steps=12):
        super().__init__()

        assert in_steps >= out_steps

        self.in_steps = in_steps
        self.out_steps = out_steps

        self.placeholder = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, T, N, C)

        self.placeholder.data[0] = 0 # kill backprop

        return x[:, -self.out_steps :, :, :] + self.placeholder
    
class HI_StatModel(StatModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_others()
        self.init_model()
    
    def init_others(self, dataset_name=None):
        pass

    def init_model(self, args=...):
        # load task parameters
        self.input_len = self.configs["his_len"]
        self.pred_len = self.configs["pred_len"]
        self.device = self.configs["task_device"]
        self.normalizer = self.configs["normalizer"]

        # build no-parameter model
        self.model = HistoricalInertia(in_steps=self.input_len,
                                       out_steps=self.pred_len)
        self.optimizer = self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001,
                                          weight_decay=0,
                                          eps=1e-8)
        self.criterion = nn.HuberLoss()
    
    def forward(self, x, aux_info = ...):
        x_hist = x[:, :self.input_len, :, :]
        truth = x[:, self.input_len:(self.input_len+self.pred_len), :, :]

        y_pred = self.model(x_hist)

        return y_pred, truth
    
    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_loss(self, pred, truth):
        L = self.criterion(pred, truth)
        return L

