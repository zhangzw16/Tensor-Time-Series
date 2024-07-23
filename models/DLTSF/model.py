import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
from models.model_base import MultiVarModelBase
from statsmodels.tsa.seasonal import seasonal_decompose

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]


class DLTSF_MultiVarModel(MultiVarModelBase):
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

        # load model params(loss)
        model_configs_yaml = os.path.join(os.path.dirname(__file__), "model.yml")
        model_configs = yaml.safe_load(open(model_configs_yaml))
        self.loss_name = model_configs["loss"]
        self.lr = model_configs["lr"]
        self.weight_decay = model_configs["weight_decay"]
        self.window_size = model_configs["window_size"]
        if self.loss_name == "mse":
            self.loss = nn.MSELoss()
        elif self.loss_name == "mae":
            self.loss = nn.L1Loss()
        else:  # mse loss by default
            self.loss = nn.MSELoss()
        
        #decomposition layer
        self.dcomp = series_decomp(self.window_size)
        # linear layer unit
        self.season_unit = nn.Linear(self.input_len, self.pred_len)
        self.trend_unit = nn.Linear(self.input_len, self.pred_len)
        # #weight-sharing linear unit
        # self.season_layer = nn.ModuleList()
        # self.trend_layer = nn.ModuleList()
        # for i in range(0, self.channels):
        #     self.season_layer.append(self.season_unit)
        #     self.trend_layer.append(self.trend_unit)
        # bundle up models
        self.model = nn.ModuleList()
        for i in range(0, self.channels):
            self.model.append(self.season_unit)
        for i in range(self.channels, self.channels + self.channels):
            self.model.append(self.trend_unit)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1.0e-8, weight_decay=self.weight_decay, amsgrad=False)

    def set_device(self, device):
        self.model.to(device)
        
    def forward(self, x, aux_info = ...):
        # input shape: [batch, hist+pred, dim1*dim2, 1]
        # model need: [batch, hist, dim1*dim2]
        x1 = x[:, :self.input_len,:, :]
        x_hist = x1.squeeze(-1)
        x_hist = self.normalizer.transform(x_hist)

        # decomposition
        season_in, trend_in = self.dcomp(x_hist)
        season_in = season_in.permute(0, 2, 1)
        trend_in = trend_in.permute(0, 2, 1) # size: [batch, channels, hist]
        season_out = torch.zeros([season_in.size(0), season_in.size(1), self.pred_len], dtype=season_in.dtype).to(self.device)
        trend_out = torch.zeros([trend_in.size(0), trend_in.size(1), self.pred_len], dtype=trend_in.dtype).to(self.device)

        #forward process
        for i in range(0,self.channels):
            season_out[:, i, :] = self.model[i](season_in[:, i, :])
            trend_out[:, i, :] = self.model[i + self.channels](trend_in[:, i, :])
        
        #combine
        outputs = season_out + trend_out
        outputs = outputs.permute(0,2,1) # size: [batch, pred, channels]
        outputs = self.normalizer.inverse_transform(outputs)

        #output and truth
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
        
