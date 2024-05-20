# from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
import h5py
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import json
from sklearn.preprocessing import StandardScaler
dataset_dir = '/home/ysc/workspace/Tensor-Time-Series/datasets/Tensor-Time-Series-Dataset/'

# data_dict = {
#     'ETTh1': Dataset_ETT_hour,
#     'ETTh2': Dataset_ETT_hour,
#     'ETTm1': Dataset_ETT_minute,
#     'ETTm2': Dataset_ETT_minute,
#     'custom': Dataset_Custom,
# }


# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1
#     train_only = args.train_only

#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = False
#         batch_size = args.batch_size
#         freq = args.freq
#     elif flag == 'pred':
#         shuffle_flag = False
#         drop_last = False
#         batch_size = 1
#         freq = args.freq
#         Data = Dataset_Pred
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq

#     data_set = Data(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         flag=flag,
#         size=[args.seq_len, args.label_len, args.pred_len],
#         features=args.features,
#         target=args.target,
#         timeenc=timeenc,
#         freq=freq,
#         train_only=train_only
#     )
#     print(flag, 'pair#: ',len(data_set))
#     print('input_shape:', data_set[0][0].shape)
#     print('label_shape:', data_set[0][1].shape)
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last)
#     return data_set, data_loader
def save_pickle(maker,save_path,data_name,extra_para = None):
    save_path = os.path.join(dataset_dir,'Processed_Data',data_name)
    if not os.path.exists(save_path):
    # 如果文件夹不存在，创建文件夹
        os.makedirs(save_path)
    if extra_para is not None:
        data, data_shape, type, raw_shape = maker(extra_para)
    else:
        data, data_shape, type, raw_shape = maker()
    pkl_data = {'data': data, 'data_shape': data_shape, 'data_type': type, 'raw_shape': raw_shape}
    pickle.dump(pkl_data, open(os.path.join(save_path,data_name+'.pkl'), 'wb'))
    del pkl_data['data']
    with open(os.path.join(save_path,data_name+'.json'), 'w') as f:
        json.dump(pkl_data, f ,indent=4)
    
def data_maker(args, flag):
    data_name = args.data
    if args.data == 'Metr-LA':
        maker = Metr_LA_data
    elif args.data == 'ETT_hour':
        maker = ETT_hour_data
    elif args.data == 'JONAS_NYC':
        maker = JONAS_NYC_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_taxi','taxi')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_bike','bike')
        return None
    elif args.data == 'PEMS':
        maker = PEMS_data
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'04',extra_para='PEMS04')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'08',extra_para='PEMS08')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'20',extra_para='PEMS20')

    else:
        raise ValueError('Invalid dataset')
    # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    return None

def Metr_LA_data():
    # 打开H5文件
    df = pd.read_hdf(dataset_dir+'metr-la.h5')
    raw_shape = df.shape
    # num_dim = len(data_shape)
    data = np.expand_dims(df.values, axis=-1)
    data_shape = data.shape
    data_type = 'traffic'
    return data, data_shape, data_type, raw_shape
def ETT_hour_data():
    scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(dataset_dir,'ETT','ETTh1.csv'))
    scale = False
    # if self.features == 'M' or self.features == 'MS':
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    # elif self.features == 'S':
    #     df_data = df_raw[[self.target]]

    if scale:
        scaler.fit(df_data.values)
    else:
        data = df_data.values
    raw_shape = df_data.shape
    data_type = 'energy'
    data = np.expand_dims(data, axis=-1)
    return data, data.shape, data_type, raw_shape

def JONAS_NYC_data(type):
    pack = np.load(os.path.join(dataset_dir,'JONAS-NYC','JONAS-NYC-16x8-20151024-20160131.npz'))
    if type == 'taxi':
        data = pack['taxi']
        data_type = 'traffic'
    elif type == 'bike':
        data = pack['bike']
        data_type = 'traffic'
    else:
        raise ValueError('Invalid data type')
    # meta_onehot = pack['meta_onehot']
    # tcov_relu = pack['tcov_relu']
    raw_shape = data.shape
    data = np.reshape(data,(raw_shape[0],raw_shape[1]*raw_shape[2],-1))
    return data, data.shape, data_type, raw_shape

def PEMS_data(d_type):
    if d_type == 'PEMS04':
        graph_signal_matrix_filename = os.path.join(dataset_dir,'PEMS04','pems04.npz')
        # graph_signal_matrix_filename = os.path.join(dataset_dir,'PEMS04','distance.csv')
    elif d_type == 'PEMS08':
        graph_signal_matrix_filename = os.path.join(dataset_dir,'PEMS08','pems08.npz')
    elif d_type == 'PEMS20':
        graph_signal_matrix_filename = os.path.join(dataset_dir,'PeMS20','data.csv')
    if d_type == 'PEMS20':
        data = pd.read_csv(graph_signal_matrix_filename,header=None)
        data = data.fillna(0)
        data = data.values
        raw_shape = data.shape
        data = np.expand_dims(data, axis=-1)
    else:
        # data = pd.read_csv(graph_signal_matrix_filename,header=None)
        data = np.load(graph_signal_matrix_filename)['data']
        raw_shape = data.shape
    data_type = 'traffic'
    # data = np.expand_dims(data, axis=-1)
    return data, data.shape, data_type, raw_shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pipe_sir_main: 训练与测试模型')
    # 添加参数
    # parser.add_argument('-m', '--mode', choices=['train', 'test'], required=True,
    #                     help='mode of operation')
    parser.add_argument( '--data', type=str, default='PEMS', 
                        help='config file path')
    # 解析参数
    args = parser.parse_args()
    data_maker(args, 'train', )