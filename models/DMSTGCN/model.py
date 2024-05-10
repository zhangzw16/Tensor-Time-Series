import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import yaml
import numpy as np
from models.model_base import TensorModelBase
from datasets.dataset import TTS_Dataset
from . import util

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DMSTGCN(nn.Module):
    def __init__(self, num_nodes, dropout=0.3,
                 out_dim=12, residual_channels=16, dilation_channels=16, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, days=288, dims=40, order=2, in_dim=9, normalization="batch"):
        super(DMSTGCN, self).__init__()
        skip_channels = 8
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.filter_convs_a = nn.ModuleList()
        self.gate_convs_a = nn.ModuleList()
        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a = nn.ModuleList()
        self.gconv_a = nn.ModuleList()

        self.gconv_a2p = nn.ModuleList()

        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims), requires_grad=True)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims), requires_grad=True)
        self.nodevec_a1 = nn.Parameter(torch.randn(days, dims), requires_grad=True)
        self.nodevec_a2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_a3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_ak = nn.Parameter(torch.randn(dims, dims, dims), requires_grad=True)
        self.nodevec_a2p1 = nn.Parameter(torch.randn(days, dims), requires_grad=True)
        self.nodevec_a2p2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_a2p3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_a2pk = nn.Parameter(torch.randn(dims, dims, dims), requires_grad=True)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv2d(in_channels=dilation_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a2p.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))

        self.relu = nn.ReLU(inplace=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, inputs, ind):
        """
        input: (B, F, N, T)
        """
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])
        x_a = self.start_conv_a(xo[:, [1]])
        skip = 0

        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        adp_a = self.dgconstruct(self.nodevec_a1[ind], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
        adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

        new_supports = [adp]
        new_supports_a = [adp_a]
        new_supports_a2p = [adp_a2p]

        for i in range(self.blocks * self.layers):
            # tcn for primary part
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # tcn for auxiliary part
            residual_a = x_a
            filter_a = self.filter_convs_a[i](residual_a)
            filter_a = torch.tanh(filter_a)
            gate_a = self.gate_convs_a[i](residual_a)
            gate_a = torch.sigmoid(gate_a)
            x_a = filter_a * gate_a

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            x = self.gconv[i](x, new_supports)
            x_a = self.gconv_a[i](x_a, new_supports_a)

            # multi-faceted fusion module
            x_a2p = self.gconv_a2p[i](x_a, new_supports_a2p)
            x = x_a2p + x

            # residual and normalization
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
            x = x + residual[:, :, :, -x.size(3):]
            x = self.normal[i](x)
            x_a = self.normal_a[i](x_a)

        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    
class DMSTGCN_TensorModel(TensorModelBase):
    def __init__(self, configs: dict={}):
        super().__init__(configs)
        self.configs = configs
        self.init_model()
        self.init_others()
    
    def init_model(self, args=...) -> nn.Module:
        # task config
        self.input_tensor_shape = self.configs['tensor_shape']
        self.input_len = self.configs['his_len']
        self.pred_len = self.configs['pred_len']
        self.normalizer = self.configs['normalizer']
        # 
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        # tensor shape
        # self.input_tensor_shape = model_configs['Tensor_dim'] # (170, 2)
        # IO
        # self.input_len = model_configs['input_len']
        # self.pred_len = model_configs['pred_len']
        # parameters
        self.days = model_configs['days']
        self.emb_dims = model_configs['emb_dims']
        self.conv_order = model_configs['conv_order']
        self.model_normalizer = model_configs['normalizer']
        # CNN
        self.dropout = model_configs['dropout']
        self.conv_in_dim = model_configs['conv_in_dim']
        self.residual_channels = model_configs['residual_channels']
        self.dilation_channels = model_configs['dilation_channels']
        self.end_channels = model_configs['end_channels']
        self.skip_channels = model_configs['skip_channels']
        self.kernel_size = model_configs['kernel_size']
        self.blocks = model_configs['blocks']
        self.layers = model_configs['layers']
        # Init model
        self.clip = 5
        num_nodes = self.input_tensor_shape[0]
        num_feature = self.input_tensor_shape[1]
        self.model = DMSTGCN(num_nodes, self.dropout, self.pred_len, 
                             self.residual_channels, self.dilation_channels, self.end_channels, self.kernel_size, self.blocks, self.layers, 
                             self.days, self.emb_dims, self.conv_order, self.conv_in_dim, self.model_normalizer)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=.3, patience=10, threshold=1e-3,
                                                                    min_lr=1e-5, verbose=True)
        self.criterion = util.masked_mae
    
    def init_others(self, dataset_pkl=None):
        # dataset = TTS_Dataset(self.configs['dataset_pkl'], 
        #                       self.configs['his_len'], self.configs['pred_len'], 
        #                       self.configs['test_ratio'], self.configs['valid_ratio'], 
        #                       self.configs['seed'])
        # train_index = dataset.trainset
        # trian_his_array = []
        # for idx in train_index:
        #     his, _ = dataset.get_his_pred_from_idx(idx)
        #     trian_his_array.append(his[...,0])
        # trian_his_array = np.array(trian_his_array)
        # # normalization and invserse
        # self.scaler = util.StandardScaler(mean=np.mean(trian_his_array), std=np.std(trian_his_array))
        pass

    def forward(self, x, aux_info:dict={}):
        # x [batch, time, dim1, dim2], dim1=node, dim2=feature
        # DMSGCN [batch, featrue, node, time]
        value = x[:, :, :self.input_tensor_shape[0], :self.input_tensor_shape[1]]
        value = value.permute(0,3,2,1)
        in_data = value[:, :, :, :self.input_len]
        in_data = self.normalizer.transform(in_data)
        truth = value[:, 0, :, self.input_len:self.input_len+self.pred_len]

        ind = aux_info['idxs'] % self.days
        in_data = nn.functional.pad(in_data, (1,0,0,0))
        pred = self.model(in_data, ind)

        pred = pred.transpose(1, 3)
        pred = self.normalizer.inverse_transform(pred)
        truth = torch.unsqueeze(truth, dim=1)
        return pred, truth
    
    def get_loss(self, pred, truth):
        # print(pred.shape, truth.shape);exit()
        loss = self.criterion(pred, truth, 0.0)
        return loss
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
        self.optim.step()
    
