import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import yaml
import numpy as np

from models.model_base import TensorModelBase

# from torchsummary import summary
# import warnings
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class SchemeA(nn.Module):
    def __init__(self, channels):
        super(SchemeA, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean((2, 3), keepdims=True)) / (x.var((2, 3), keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.reshape(1, -1, 1, 1, 1) + self.beta.reshape(1, -1, 1, 1, 1)
        return out
    
class SchemeB(nn.Module):
    def __init__(self, channels):
        super(SchemeB, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean((2, 4), keepdims=True)) / (x.var((2, 4), keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.reshape(1, -1, 1, 1, 1) + self.beta.reshape(1, -1, 1, 1, 1)
        return out
    
class SchemeC(nn.Module):
    def __init__(self, channels):
        super(SchemeC, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean((3, 4), keepdims=True)) / (x.var((3, 4), keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.reshape(1, -1, 1, 1, 1) + self.beta.reshape(1, -1, 1, 1, 1)
        return out

class SchemeD(nn.Module):
    def __init__(self, num_nodes, channels):
        super(SchemeD, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1,channels,1,num_nodes,1))
        self.gamma = nn.Parameter(torch.ones(1,channels,1,num_nodes,1))

    def forward(self, x):
        mean = (x - x.mean(2, keepdims=True))
        var = x.var(2, keepdims=True, unbiased=True)
        var = torch.where(torch.isnan(var), torch.full_like(var, 0), var)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out

class SchemeE(nn.Module):
    def __init__(self, channels):
        super(SchemeE, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(3, keepdims=True)) / (x.var(3, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.reshape(1, -1, 1, 1, 1) + self.beta.reshape(1, -1, 1, 1, 1)
        return out
    
class SchemeF(nn.Module):
    def __init__(self, num_nodes, num_source, channels, track_running_stats=True, momentum=0.1):
        super(SchemeF, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_source, int(num_nodes), 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_source, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_source, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_source, num_nodes, 1))
        self.momentum = momentum        
        
    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 4), keepdims=True)
            var = x.var((0, 4), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[4] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((4), keepdims=True)
            var = x.var((4), keepdims=True, unbiased=True)

        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out

class ResidualBlock(nn.Module):
    def __init__(self, num_nodes, num_source, n_pred, n_his, channels, dilation, kernel_size, schemeA_bool, schemeB_bool, schemeC_bool, schemeD_bool, schemeE_bool, schemeF_bool, fusion_bool):
        super(ResidualBlock, self).__init__()
        self.num_source = num_source
        self.dilation = dilation
        self.dilation_channels = channels
        self.schemeA_bool = schemeA_bool
        self.schemeB_bool = schemeB_bool
        self.schemeC_bool = schemeC_bool
        self.schemeD_bool = schemeD_bool
        self.schemeE_bool = schemeE_bool
        self.schemeF_bool = schemeF_bool        
        self.fusion_bool = fusion_bool
        
        if self.schemeA_bool:
            self.schemeA = SchemeA(channels)
        if self.schemeB_bool:
            self.schemeB = SchemeB(channels)
        if self.schemeC_bool:
            self.schemeC = SchemeC(channels)
        if self.schemeD_bool:
            self.schemeD = SchemeD(num_nodes, channels)
        if self.schemeE_bool:
            self.schemeE = SchemeE(channels)
        if self.schemeF_bool:
            self.schemeF = SchemeF(num_nodes, num_source, channels)             

        num_oneway = int(self.schemeA_bool) + int(self.schemeB_bool) + int(self.schemeC_bool) 
        num_twoway = int(self.schemeD_bool) + int(self.schemeE_bool) + int(self.schemeF_bool)
        num = num_oneway + num_twoway + 1
        
        if self.fusion_bool:
            self.fusion = nn.Bilinear((num_oneway+1) * channels, (num_twoway+1) * channels, num * channels)
        self.filter_convs = nn.Conv3d(in_channels = num * channels, 
                                      out_channels = num_source * self.dilation_channels, 
                                      kernel_size = (num_source, 1, kernel_size),
                                      dilation=(1,1,self.dilation))
        self.gate_convs = nn.Conv3d(in_channels = num * channels, 
                                    out_channels = num_source * self.dilation_channels, 
                                    kernel_size = (num_source, 1, kernel_size),
                                    dilation=(1,1,self.dilation))
        self.residual_convs = nn.Conv3d(in_channels = channels, 
                                        out_channels = channels, 
                                        kernel_size = (1,1,1))
        self.skip_convs = nn.Conv3d(in_channels = channels, 
                                    out_channels = channels, 
                                    kernel_size = (1,1,1))

    def forward(self, x):
        residual = x
        x1_list = []
        x2_list = []
        if self.schemeA_bool:
            x_schemeA = self.schemeA(x)
            x1_list.append(x_schemeA)
        if self.schemeB_bool:
            x_schemeB = self.schemeB(x)
            x1_list.append(x_schemeB)
        if self.schemeC_bool:
            x_schemeC = self.schemeC(x)
            x1_list.append(x_schemeC)
        if self.schemeD_bool:
            x_schemeD = self.schemeD(x)
            x2_list.append(x_schemeD)    
        if self.schemeE_bool:
            x_schemeE = self.schemeE(x)
            x2_list.append(x_schemeE)
        if self.schemeF_bool:
            x_schemeF = self.schemeF(x)
            x2_list.append(x_schemeF)
        if self.fusion_bool:     
            x1_list.append(x)
            x2_list.append(x)
            x1 = torch.cat(x1_list, dim=1)
            x2 = torch.cat(x2_list, dim=1)
            x_f = self.fusion(x1.contiguous().permute(0,2,3,4,1), x2.contiguous().permute(0,2,3,4,1))
            x = x_f.permute(0,4,1,2,3)
        else:
            x2_list.append(x)
            x1 = torch.cat(x1_list, dim=1)
            x2 = torch.cat(x2_list, dim=1)
            x = torch.cat((torch.cat(x1_list, dim=1), torch.cat(x2_list, dim=1)), dim=1)

        filter = self.filter_convs(x)
        b, _, _, n, t = filter.shape
        filter = torch.tanh(filter).reshape(b, -1, self.num_source, n, t)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate).reshape(b, -1, self.num_source, n, t)
        x = filter * gate
        # parametrized skip connection
        save_x = x
        sk = x
        sk = self.skip_convs(sk)
        x = self.residual_convs(x)
        return x, sk, gate
    
class TTSNorm(nn.Module):
    def __init__(self,  num_nodes, num_source, n_his, n_pred, schemeA_bool=False, schemeB_bool=False, schemeC_bool=False, schemeD_bool=False, 
                 schemeE_bool=False, schemeF_bool=False, in_dim=1, out_dim=1, channels=32, kernel_size=2,layers=2, fusion_bool=False):
        super(TTSNorm, self).__init__()
        self.num_source = num_source
        self.layers = layers
        self.in_dim = in_dim        
        self.residualblocks = nn.ModuleList()
        self.start_conv = nn.Conv3d(in_channels = in_dim, 
                                    out_channels = channels, 
                                    kernel_size = (1,1,1))
        dilation = 1
        for i in range(layers):
            self.residualblocks.append(ResidualBlock(num_nodes, num_source, n_pred, n_his, channels, dilation, kernel_size, 
                                                     schemeA_bool, schemeB_bool, schemeC_bool, schemeD_bool, schemeE_bool, schemeF_bool, fusion_bool))
            dilation *= 2
        self.end_conv_1 = nn.Conv3d(in_channels = channels,
                                    out_channels = channels,
                                    kernel_size = (1,1,1),
                                    bias = True)
        self.end_conv_2 = nn.Conv3d(in_channels = channels, 
                                    out_channels = n_pred * out_dim, 
                                    kernel_size = (1,1,1), 
                                    bias=True)


    def forward(self, input):
        input = input.permute(0, 4, 3, 2, 1) # [b,t,n,s,c] - > [b,c,s,n,t]
        x = self.start_conv(input)
        skip = 0
        
        for i in range(self.layers):
            residual = x
            x, sk, gate = self.residualblocks[i](x)
            x = x + residual[:, :, :, :, -x.size(4):]
            try:
                skip = sk + skip[:, :, :, :, -sk.size(4):]
            except:
                skip = sk
        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(rep)
        return out.permute(0, 1, 3, 2, 4) 

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)

class TTS_Norm_TensorModel(TensorModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.init_model()
        self.init_others()

    def init_model(self, args=...) -> nn.Module:
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        self.tensor_shape = model_configs['Tensor_dim']
        self.input_len = model_configs['input_len']
        self.pred_len  = model_configs['pred_len']
        self.fusion = model_configs['fusion']
        self.mix_loss = model_configs['mix_loss']
        self.hidden_channels = model_configs['hidden_channels']
        self.scaler = model_configs['scaler']
        self.schemeA = model_configs['schemeA']
        self.schemeB = model_configs['schemeB']
        self.schemeC = model_configs['schemeC']
        self.schemeD = model_configs['schemeD']
        self.schemeE = model_configs['schemeE']
        self.schemeF = model_configs['schemeF']
        
        self.layer = int(np.log2(self.input_len))

        self.model = TTSNorm(num_nodes=self.tensor_shape[0], num_source=self.tensor_shape[1], n_his=self.input_len, n_pred=self.pred_len, 
                             schemeA_bool=self.schemeA, schemeB_bool=self.schemeB, schemeC_bool=self.schemeC,
                             schemeD_bool=self.schemeD, schemeE_bool=self.schemeE, schemeF_bool=self.schemeF,
                             in_dim=1, out_dim=1, channels=self.hidden_channels, kernel_size=2, layers=self.layer, fusion_bool=self.fusion)
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)
        # filter(lambda p: p.requires_grad, self.model.parameters())
        self.criterion = nn.MSELoss()

    def init_others(self, dataset_name=None):
        pass
    
    def forward(self, x, aux_info: dict = ...):
        # x [batch, time, dim1, dm2]
        # TTS_Norm: (batch, n_his+n_pred, dim1, dim2, 1)
        value = x[:, :, :self.tensor_shape[0], :self.tensor_shape[1]].unsqueeze(-1)
        in_data = value[:, :self.input_len, :, :, :]
        truth = value[:, self.input_len:self.input_len+self.pred_len, :, :, :]
        # self.model.zero_grad()
        pred = self.model(in_data)
        return pred, truth
    
    def backward(self, loss):
        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optim.step()
    
    def get_loss(self, pred, truth):
        if self.mix_loss:
            var = torch.var(pred, unbiased=True)
            shrinkage = 1 - var / torch.sum(pred**2)
            theta = shrinkage * pred
            loss = self.criterion(theta, truth)
        else:
            loss = self.criterion(pred, truth)
        return loss
    
    def set_device(self, device='cpu'):
        self.model.to(device)