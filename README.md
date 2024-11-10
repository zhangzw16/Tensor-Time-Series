# Tensor-Time-Series-Library
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
[![Visits Badge](https://badges.pufler.dev/visits/zhangzw16/Tensor-Time-Series)](https://github.com/zhangzw16/Tensor-Time-Series)
[![Updated Badge](https://badges.pufler.dev/updated/zhangzw16/Tensor-Time-Series)](https://github.com/zhangzw16/Tensor-Time-Series)


Papers and datasets for tensor time series.

## Papers with Code

- `TTS-Norm` TTS-Norm: Forecasting Tensor Time Series via Multi-Way Normalization (ACM TKDD 2023) [[paper](https://dl.acm.org/doi/10.1145/3605894)] [[code](https://github.com/beginner-sketch/TTSNorm)]
- `GMRL` Learning Gaussian Mixture Representations for Tensor Time Series Forecasting (IJCAI 2023) [[paper](https://www.ijcai.org/proceedings/2023/0231.pdf)] [[code](https://github.com/beginner-sketch/GMRL)]
- `NET3` Network of Tensor Time Series (WWW 2021) [[paper](https://arxiv.org/abs/2102.07736)] [[code](https://github.com/baoyujing/NET3)]
- `STC-GNN` Spatio-Temporal-Categorical Graph Neural Networks for Fine-Grained Multi-Incident Co-Prediction (CIKM 2021) [[paper](https://dl.acm.org/doi/abs/10.1145/3459637.3482482)] [[code](https://github.com/underdoc-wang/STC-GNN)]
- `DMSTGCN` Dynamic and multi-faceted spatiotemporal deep learning for traffic speed forecasting (KDD 2021) [[paper](https://dl.acm.org/doi/10.1145/3447548.3467275)] [[code](https://github.com/liangzhehan/DMSTGCN/tree/main)]
Here is the list of papers organized in the requested format:
- `ST-Norm` Spatial and temporal normalization for multi-variate time series forecasting (KDD 2021) [[paper](https://dl.acm.org/doi/10.1145/3447548.3467330)] [[code](https://github.com/JLDeng/ST-Norm)]
- `MTGNN` Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks (NeurIPS 2020) [[paper](https://arxiv.org/abs/2005.11650)] [[code](https://github.com/nnzhan/MTGNN)]
- `AGCRN` Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting (NeurIPS 2020) [[paper](https://proceedings.neurips.cc/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf)] [[code]()]
- `StemGNN` Spectral temporal graph neural network for multivariate time-series forecasting (NeurIPS 2020) [[paper](https://arxiv.org/abs/2103.07719)] [[code](https://github.com/microsoft/StemGNN)]
- `Graph Wavenet` Graph WaveNet for Deep Spatial-Temporal Graph Modeling (IJCAI 2019) [[paper](https://arxiv.org/abs/1906.00121)] [[code](https://github.com/nnzhan/Graph-WaveNet)]
- `DCRNN` Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (ICLR 2018) [[paper](https://arxiv.org/abs/1707.01926)] [[code](https://github.com/liyaguang/DCRNN)]
- `STGCN` Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting (IJCAI 2018) [[paper](https://arxiv.org/abs/1709.04875)] [[code](https://github.com/VeritasYin/STGCN_IJCAI-18)]
- `GCGRU`  [[code](https://github.com/XDZhelheim/Torch-MTS)]
- `GCLSTM` GC-LSTM: Graph Convolution Embedded LSTM for Dynamic Link Prediction (Applied Intelligence 2022) [[paper](https://arxiv.org/abs/1812.04206)] [[code](https://github.com/XDZhelheim/Torch-MTS)]
- `HI` Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting (CIKM 2021) [[paper](https://dl.acm.org/doi/abs/10.1145/3459637.3482120)] [[code](https://github.com/XDZhelheim/Torch-MTS)]
- `STID` Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting (CIKM 2022) [[paper](https://arxiv.org/abs/2208.05233)] [[code](https://github.com/XDZhelheim/Torch-MTS)]
- `STWA` Towards Spatio-Temporal Aware Traffic Time Series Forecasting (ICDE 2022) [[paper](https://arxiv.org/abs/2203.15737)] [[code](https://github.com/razvanc92/ST-WA)]
- `Mamba` Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2312.00752)] [[code](https://github.com/XDZhelheim/Torch-MTS)]
- `TimesNet` TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (ICLR 2023) [[paper](https://arxiv.org/abs/2210.02186)] [[code](https://github.com/thuml/Time-Series-Library)]



## Datasets

For datasets, please refer to [Datasets](./datasets/README.md)

## Forecast Paradigms
<!-- insert imgs -->
![forecast_paradigms](./imgs/forecast_paradigms.png)

## Get Started

Create a virtual environment before we get stated. (Python >= 3.8)

```shell
conda create --name TensorTSL
```

An easy way to install the environment is to use `pip install` with the config file `pyproject.toml`. 

```shell
pip install .
```


## Have a try

### 1. run a simple task
We provide some scripts to run tasks easily.

```shell
# you can run by using python3 directly.
python3 main.py
# or you can choose the command line version
python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --dataset_base $dataset_base\
                    --scheduler $scheduler \
                    --lr_finder \
                    --logger $logger \
                    --dataset $dataset --model $model \
# isn't it this a little complicated? We also provide some relatviely simple version of shell stripts in './bash_scripts/'
bash ./bash_scripts/run.sh
# you can find more details in the run.sh
```

### 2. run tasks

Based on `main_cli.py` and `run.sh`, we also provide some scripts for you in `bash_scripts/*.sh`. All these scripts are easly to understand and cutomize.

```shell
# run a model across datasets in different domains
# all parameters are required
bash bash_scripts/run_traffic.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --graph_init $graph_init --data_mode $data_mode
bash bash_scripts/run_finance.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --graph_init $graph_init --data_mode $data_mode
bash bash_scripts/run_weather.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --graph_init $graph_init --data_mode $data_mode
bash bash_scripts/run_nature.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --graph_init $graph_init --data_mode $data_mode
bash bash_scripts/run_energy.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --graph_init $graph_init --data_mode $data_mode
```

> [!TIP]
> The scripts shown above only run **a model** once.
> If you want to run different model, please set the variable `$model` before you get started.

