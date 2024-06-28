# TensorModel - STC-GNN
**Title**: Spatio-Temporal-Categorical Graph Neural Networks for Fine-Grained Multi-Incident Co-Prediction

**Authors**: Zhaonan Wang, Renhe Jiang, Zekun cai, Zipei Fan

**Comments**： Accepted by CIKM'2021 

**Paper**: http://dx.doi.org/10.1145/3459637.3482482

**Github**: https://github.com/underdoc-wang/STC-GNN

# Work Recurrence

Maintained by ChongKaKam - zhuangjx23@mails.tsinghua.edu.cn

+ finished basically except the graph As,Ac

+ data shape: (5124, 10, 10, 5) = (time, grid1, grid2, categories)

+ n_his = 9, n_pred = 3

+ input tensor shape: (batch_size, time_step, num_grids, categories) where num_grids = (H x W), categories = C. The H,W,C are related to the dataset -- (32, 9, 100, 5) (32, 3, 100, 5)

+ The initialization of graph As and graph Ac is related to the dataset.
