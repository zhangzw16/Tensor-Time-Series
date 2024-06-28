# TensorModel - DMSTGCN

**Title**: Dynamic and multi-faceted spatiotemporal deep learning for traffic speed forecasting

**Authors**: Liangzhen Han, Bowen Du, Leilei Sun...

**Comments**： Accepted by KDD'21

**Paper**: https://dl.acm.org/doi/10.1145/3447548.3467275

**Github**: https://github.com/liangzhehan/DMSTGCN/tree/main

# Work Recurrence

Maintained by ChongKaKam - zhuangjx23@mails.tsinghua.edu.cn

+ status: completed basiclly. 

+ data shape: (10699, 12, 170, 2) in PEMSD8

+ model configs:

+ input tensor shape: (batch_size, featrue, node_num, time). PS: feature=2, time=input_len=12, node_num is related to datasets.

+ output tensor shape: (batch, pred_len, node_num, out_features)

+ Attention: the dimension of featrues in input tensor is 2, the first feature is target, and the second feature is auxiliary. Therefore, the x shape = (batch, time, node, 0:2), y shape = (batch, time, node, 0:1)
