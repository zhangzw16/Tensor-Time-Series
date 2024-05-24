import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import scipy.sparse as sp
import sklearn.preprocessing as preprocessing
import numpy as np
import yaml
import os

from .layers import layers
from .script import earlystopping, utility
from models.model_base import MultiVarModelBase



class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self,
        n_his,
        Kt,
        Ks,
        act_func,
        graph_conv_type,
        gso,
        enable_bias,
        droprate,
        blocks,
        n_vertex,
    ):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                layers.STConvBlock(
                    Kt,
                    Ks,
                    n_vertex,
                    blocks[l][-1],
                    blocks[l + 1],
                    act_func,
                    graph_conv_type,
                    gso,
                    enable_bias,
                    droprate,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = n_his - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(
                Ko,
                blocks[-3][-1],
                blocks[-2],
                blocks[-1][0],
                n_vertex,
                act_func,
                enable_bias,
                droprate,
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=enable_bias
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0], out_features=blocks[-1][0], bias=enable_bias
            )
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x


class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self,
        n_his,
        Kt,
        Ks,
        act_func,
        graph_conv_type,
        gso,
        enable_bias,
        droprate,
        blocks,
        n_vertex,
    ):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                layers.STConvBlock(
                    Kt,
                    Ks,
                    n_vertex,
                    blocks[l][-1],
                    blocks[l + 1],
                    act_func,
                    graph_conv_type,
                    gso,
                    enable_bias,
                    droprate,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = n_his - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(
                Ko,
                blocks[-3][-1],
                blocks[-2],
                blocks[-1][0],
                n_vertex,
                act_func,
                enable_bias,
                droprate,
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=enable_bias
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0], out_features=blocks[-1][0], bias=enable_bias
            )
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.do = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x


class STGCN_MultiVarModel(MultiVarModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs  # task configs
        self.init_others()
        self.init_model()

    # # original method implemented in STGCN, use extra spatial info
    # def _load_adj(self, dataset: str):
    #     dataset_name = "./" + dataset
    #     adj_path = os.path.join(dataset_name, "adj.npz")
    #     adj = sp.load_npz(adj_path)
    #     adj = adj.tocsc()

    #     if dataset == "metr-la":
    #         n_vertex = 207
    #     elif dataset == "pems-bay":
    #         n_vertex = 325
    #     elif dataset == "pemsd7-m":
    #         n_vertex = 228

    #     return adj, n_vertex

    def init_model(self, args=...):
        # load task parameters
        self.input_len = self.configs["his_len"]
        self.pred_len = self.configs["pred_len"]
        self.device = self.configs["task_device"]

        # use graph generator implemented in utils.graph
        self.normalizer = preprocessing.StandardScaler() #self.configs["normalizer"]

        # configure model parameters
        model_configs_yaml = os.path.join(os.path.dirname(__file__), "model.yml")
        model_configs = yaml.safe_load(open(model_configs_yaml))
        self.gso_type = model_configs["gso_type"]
        self.Kt = 3  # model_configs['Kt']
        self.stblock_num = model_configs["stblock_num"]
        self.act_func = "glu"  # model_configs["act_func"]
        self.Ks = 2  # model_configs['Ks']
        self.graph_conv_type = model_configs["graph_conv_type"]
        self.enable_bias = True  # model_configs['enable_bias']
        self.droprate = model_configs["droprate"]
        self.lr = model_configs["lr"]
        self.weight_decay_rate = model_configs["weight_decay_rate"]
        self.batch_size = model_configs["batch_size"]
        self.epochs = model_configs["epochs"]
        self.opt = "adam"  # model_configs['opt']
        self.step_size = 10  # model_configs['step_size']
        self.gamma = 0.95  # model_configs["gamma"]
        self.patience = 30  # model_configs["patience"]

        self.Ko = self.input_len - (self.Kt - 1) * 2 * self.stblock_num
        gso = utility.calc_gso(self.adj, self.gso_type)
        if self.graph_conv_type == "cheb_graph_conv":
            gso = utility.calc_chebynet_gso(gso)
        gso = gso.toarray()
        gso = gso.astype(dtype=np.float32)
        self.gso = torch.from_numpy(gso).to(self.device)

        # blocks: settings of channel size in st_conv_blocks and output layer,
        # using the bottleneck design in st_conv_blocks
        self.blocks = []
        self.blocks.append([1])
        for l in range(self.stblock_num):
            self.blocks.append([64, 16, 64])
        if self.Ko == 0:
            self.blocks.append([128])
        elif self.Ko > 0:
            self.blocks.append([128, 128])
        self.blocks.append([1])

        # load model
        self.loss = nn.MSELoss()
        self.es = earlystopping.EarlyStopping(
            mode="min", min_delta=0.0, patience=self.patience
        )
        if self.graph_conv_type == "cheb_graph_conv":
            self.model = STGCNChebGraphConv(
                self.input_len,
                self.Kt,
                self.Ks,
                self.act_func,
                self.graph_conv_type,
                self.gso,
                self.enable_bias,
                self.droprate,
                self.blocks,
                self.n_vertex,
            )
        else:
            self.model = STGCNGraphConv(
                self.input_len,
                self.Kt,
                self.Ks,
                self.act_func,
                self.graph_conv_type,
                self.gso,
                self.enable_bias,
                self.droprate,
                self.blocks,
                self.n_vertex,
            )

        if self.opt == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay_rate
            )
        elif self.opt == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay_rate,
                amsgrad=False,
            )
        elif self.opt == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay_rate,
                amsgrad=False,
            )
        else:
            raise NotImplementedError(
                f"ERROR: The optimizer {self.opt} is not implemented."
            )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

    def init_others(self, args=...):
        # data shape: (time, dim1*dim2,1), multivariate data
        #                    0         1
        self.graph_generator = self.configs['graphGenerator']
        self.adj = torch.from_numpy(
            self.graph_generator.cosine_similarity_matrix(n_dim=0, normal=True)
        ).float()
        self.n_vertex = self.adj.size(0)

    def set_device(self, device):
        self.model.to(device)

    def forward(self, x, aux_info=...):
        # dataloader: [batch_size, time(hist+pred), dim1*dim2(n_vertex), 1]
        #   x=[batch_size, 1, n_his, n_vertex]
        #   y=[batch_size, n_vertex]
        x1 = x[:, : self.input_len, :, :]
        x_hist = x1.permute(0, 3, 1, 2)
        y = x[:, self.input_len + self.pred_len - 1, :, :]
        truth = y.squeeze()
        y_pred = self.model(x_hist).view(len(x), -1)  # y_pred: [batch_size, n_vertex]
        return y_pred, truth

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_loss(self, pred, truth):
        L = self.loss(pred, truth)
        return L


# if __name__ == "__main__":
#     configs = {"dataset": "metr-la", "device": "cuda", "his_len": 12, "pred_len": 3}
#     model_default_configs = {
#         "gso_type": "sym_norm_lap",
#         "stblock_num": 2,
#         "graph_conv_type": "cheb_graph_conv",
#         "droprate": 0.5,
#         "lr": 0.001,
#         "weight_decay_rate": 0.0001,
#         "batch_size": 32,
#         "epochs": 100,
#     }
#     m = STGCN_TensorModel(configs=configs)
#     input=torch.rand(32, 15, 207, 1).to(m.device)
#     input=input.to(m.device)
#     pred, truth = m.forward(input)
#     print(pred.shape, truth.shape)
#     loss=m.get_loss(pred, truth)
#     print(loss.item())
#     m.backward(m.get_loss(pred, truth))
