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
import yaml
from tsf_loader import convert_tsf_to_dataframe
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
        data, data_shape, type, raw_shape, resolution = maker(extra_para)
    else:
        data, data_shape, type, raw_shape, resolution = maker()
    pkl_data = {'data': data, 'data_shape': list(data_shape), 'data_type': type, 'raw_shape': list(raw_shape), 'temporal_resolution': resolution}
    pickle.dump(pkl_data, open(os.path.join(save_path,data_name+'.pkl'), 'wb'))
    del pkl_data['data']
    with open(os.path.join(save_path,data_name+'.yaml'), 'w') as f:
        yaml.dump(pkl_data, f ,indent=4)
    
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
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'03',extra_para='PEMS03')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'04',extra_para='PEMS04')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'07',extra_para='PEMS07')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'08',extra_para='PEMS08')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'20',extra_para='PEMS20')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'BAY',extra_para='PEMS_BAY')
    elif args.data == 'METRO':
        maker = Metro_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_SH',extra_para='SHMETRO')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_HZ',extra_para='HZMETRO')
    elif args.data == 'COVID':
        maker = COVID_data
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_CHI','COVID-CHI')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_US','COVID-US')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_DEATHS','COVID-DEATHS')

    else:
        raise ValueError('Invalid dataset')
    # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    return None
def COVID_data(d_type):
    if d_type == 'COVID-CHI':
        signal = pd.read_csv(os.path.join(dataset_dir,'COVID-CHI','data.csv'))
        print(signal)
        data = signal.values
        data = data[:,1:]
        raw_shape = data.shape
        data = np.reshape(data,(raw_shape[0],14,-1))
        data_type = 'transport'
        resolution = '2hours'
        
    elif d_type == 'COVID-US':
        signal = np.load(os.path.join(dataset_dir,'COVID-US','COVID-US-51x1-20191114-20200531.npz'))
        data_type = 'travel'
        data = signal['poi']
        raw_shape = data.shape
        resolution = '1hour'

    elif d_type == 'COVID-DEATHS':
        path = os.path.join(dataset_dir,'COVID-DEATHS','covid_deaths_dataset.tsf')
        data,frequency,forecast_horizon,contain_missing_values,contain_equal_length, = convert_tsf_to_dataframe(path)
        data = data['series_value']
        fetch_data = []
        for i in range(len(data)):
            fetch_data.append(data[i].tolist())
        data = np.array(fetch_data).T
        raw_shape = data.shape
        data = np.expand_dims(data, axis=-1)
        resolution = '1day'
        data_type = 'natural'
    data_shape = data.shape
    return data, data_shape, data_type, raw_shape, resolution
def Metro_data(d_type):
    if d_type == 'SHMETRO':
        signal = os.path.join(dataset_dir,'SHMETRO','SHMETRO.dyna')
    elif d_type == 'HZMETRO':
        signal = os.path.join(dataset_dir,'HZMETRO','HZMETRO_new.dyna')
    else:
        raise ValueError('Invalid data type')
    df = pd.read_csv(signal)
    print(df)
    # Ensure the time column is of datetime type
    df['time'] = pd.to_datetime(df['time'])
    # Sort the DataFrame by 'time' and 'entity_id'
    df_sorted = df.sort_values(by=['time', 'entity_id'])
    # Pivot the DataFrame to get 'time' as index, 'entity_id' as columns, and 'traffic_flow' as values
    in_flow_pivot = df_sorted.pivot(index='time', columns='entity_id', values='inflow')
    out_flow_pivot = df_sorted.pivot(index='time', columns='entity_id', values='outflow')
    # Reset the index if needed
    in_flow_pivot = out_flow_pivot.reset_index()
    out_flow_pivot = out_flow_pivot.reset_index()
    print(in_flow_pivot)
    print(out_flow_pivot)
    in_flow_pivot = in_flow_pivot.drop(columns=['time'])
    out_flow_pivot = out_flow_pivot.drop(columns=['time'])
    # Display the new DataFram
    data = np.array([in_flow_pivot.values, out_flow_pivot.values]).transpose(1,2,0)
    raw_shape = data.shape
    # data = np.expand_dims(data, axis=-1)
    data_type = 'traffic'
    data_resolution = '15mins'
    return data, data.shape, data_type, raw_shape, data_resolution
    
def Metr_LA_data():
    # 打开H5文件
    df = pd.read_hdf(dataset_dir+'metr-la.h5')
    raw_shape = df.shape
    # num_dim = len(data_shape)
    data = np.expand_dims(df.values, axis=-1)
    data_shape = data.shape
    data_type = 'traffic'
    resolution = '5mins'
    return data, data_shape, data_type, raw_shape, resolution
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
    resolution = '1hour'
    return data, data.shape, data_type, raw_shape, resolution

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
    data_resolution = '30minus'
    return data, data.shape, data_type, raw_shape, data_resolution

def PEMS_data(d_type):
    if d_type == 'PEMS03':
        # signal = os.path.join(dataset_dir,'PEMS04','pemsD4.npz')
        signal = os.path.join(dataset_dir,'PEMS03','PEMSD3.dyna')
        # data = np.load(signal)['data']
        # raw_shape = data.shape
        # graph_signal_matrix_filename = os.path.join(dataset_dir,'PEMS04','distance.csv')
    elif d_type == 'PEMS04':
        signal = os.path.join(dataset_dir,'PEMSD4','PEMSD4.dyna')
    elif d_type == 'PEMS07':
        signal = os.path.join(dataset_dir,'PEMSD7','PEMSD7.dyna')
    elif d_type == 'PEMS08':
        signal = os.path.join(dataset_dir,'PEMSD8','PEMSD8.dyna')
        # data = np.load(signal)['data']
        # raw_shape = data.shape
    elif d_type == 'PEMS20':
        signal = os.path.join(dataset_dir,'PeMS20','data.csv')
        data = pd.read_csv(signal,header=None)
        data = data.fillna(0)
        data = data.values
        raw_shape = data.shape
        data = np.expand_dims(data, axis=-1)
    elif d_type == 'PEMS_BAY':
        df = pd.read_hdf(dataset_dir+'PEMS_BAY/pems-bay.h5')
        raw_shape = df.shape
        # num_dim = len(data_shape)
        data = np.expand_dims(df.values, axis=-1)
        # data_shape = data.shape
    if d_type == 'PEMS20' or d_type == 'PEMS_BAY':
        data_type = 'traffic'
    else:
        df = pd.read_csv(signal)
        # Ensure the time column is of datetime type
        df['time'] = pd.to_datetime(df['time'])
        # Sort the DataFrame by 'time' and 'entity_id'
        df_sorted = df.sort_values(by=['time', 'entity_id'])
        # Pivot the DataFrame to get 'time' as index, 'entity_id' as columns, and 'traffic_flow' as values
        traffic_flow_pivot = df_sorted.pivot(index='time', columns='entity_id', values='traffic_flow')
        # Reset the index if needed
        traffic_flow_pivot = traffic_flow_pivot.reset_index()
        print(traffic_flow_pivot)
        traffic_flow_pivot = traffic_flow_pivot.drop(columns=['time'])
        # Display the new DataFram
        data = traffic_flow_pivot.values
        raw_shape = data.shape
        data = np.expand_dims(data, axis=-1)
    data_type = 'traffic'
    data_resolution = '5mins'
    # data = np.expand_dims(data, axis=-1)
    return data, data.shape, data_type, raw_shape, data_resolution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pipe_sir_main: 训练与测试模型')
    # 添加参数
    # parser.add_argument('-m', '--mode', choices=['train', 'te], required=True,
    #                     help='mode of operation')
    parser.add_argument( '--data', type=str, default='METRO', 
                        help='config file path')
    # 解析参数
    args = parser.parse_args()
    data_maker(args, 'train', )