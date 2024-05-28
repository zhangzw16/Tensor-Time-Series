# Models

There are two types of models, 'Tensor' or 'MultiVar'.

Tensor model supports (batch, time, dim1, dim2) input and output, while Multivariate model only supports (batch, time, dim1) input and output.

## Tensor Model (8)
+ TTS_Norm: Normalization - (ACM Trans 2023)
+ GMRL:  Gaussian  - (IJCAI 2023)
+ !!! MegaCRN: (AAAI 2023)
+ !!! STGM: G - (Expert Systems with Applications 2023)

+ NET3:  GCN       (prior graph) - (WWW 2021)
+ DCRNN: GNN       (prior graph) - (ICLR 2018)
+ GWNet: GCN + RNN (prior graph + update)
+ !!! DGCRN

+ AGCRN: GCN + RNN (learned graph) - (NeurIPS 2020)
+ MTGNN: GCN + TCN (learned graph) - (KDD 2020)

## Multivariate Model (7)
+ TimesNet: Embedding + Inception - (ICLR 2023)
+ ST_Norm:  Normalization         - (KDD 2021)
+ StemGNN:  GCN + TCN + Spectral  - (NeurIPS 2020)
+ STGCN: GCN + CNN (prior graph)
+ AutoFormer: Transformer-based - ()
+ !!! patchTST  (LLM)
+ !!! TTM   (fundation model)

## Statistic Model (1)
+ HM