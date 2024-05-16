# Models

There are two types of models, 'Tensor' or 'MultiVar'.

Tensor model supports (batch, time, dim1, dim2) input and output, while Multivariate model only supports (batch, time, dim1) input and output.

## Tensor Model (6)
+ TTS_Norm: Normalization - (ACM Trans 2023)
+ GMRL:  Gaussian  - (IJCAI 2023)
+ NET3:            - (WWW 2021)
<!-- + STC_GNN:         - (CIKM 2021) -->
+ DMSTGCN:         - (KDD 2021)
+ AGCRN: GCN + RNN - (NeurIPS 2020)
+ MTGNN: GCN + TCN - (KDD 2020)


## Multivariate Model (3)
<!-- + GTS:     GCN + RNN            - (ICLR 2021) -->
+ TimesNet: Embedding + Inception - (ICLR 2023)
+ ST_Norm:  Normalization         - (KDD 2021)
+ StemGNN:  GCN + TCN + Spectral  - (NeurIPS 2020)


## TODO: Statistic Model (2)
+ HM
+ ARIMA