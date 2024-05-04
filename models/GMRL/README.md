# TensorModel - GMRL

**Title**: Learning Gaussian Mixture Representations for Tensor Time Series Forecasting

**Authors**: Jiewen Deng, Jinliang Deng, Renhe Jiang, Xuan Song

**Comments**： Accepted by IJCAI 2023 Main Track

**Paper**: https://arxiv.org/abs/2306.00390

**Github**: https://github.com/beginner-sketch/GMRL

# Work Recurrence

Maintained by ChongKaKam - zhuangjx23@mails.tsinghua.edu.cn

+ status: complete. 

+ data shape (4368, 98, 4). There are 98 locations with 4 types of flow (source) and the total timesteps are 4368

+ defualt n_input = 16, n_pred = 3. The total length of a sample is 19

+ input tensor shape: (batch_size, time_step, tensor_dim1, tensor_dim2, 1)

+ model input args:
    ```python
    '''
    Model: GRML
    Args:
        - device: 'cpu' or 'cuda'
        - num_comp: number of Gaussian components
        - num_nodes: the first dimension of the input Tensor (98 in this NYC case represents the 98 lacations)
        - num_source: the second dimension of the input Tensor (4 in this NYC case represents the 4 sources [Bike, Taxi]x[Inflow, Outflow])
        - n_his: the time steps of input seq
        - n_pred: the time steps of prediction seq
        - in_dim: input dimension is 1 (sequence)
        - out_dim: output dimension is 1 (sequence)
        - channels: the hidden channels
        - kernal_size: 2 in default
        - layers: int(log_2(n_his))
        - hra_bool: using HRA or not
    '''
    GMRL(device=device, num_comp=num_comp, num_nodes=num_nodes, num_source=num_source, n_his=n_his, n_pred=n_pred, in_dim=1, out_dim=1, channels=hidden_channels, kernel_size=2, layers=layers, hra_bool=hra_bool)
    ```