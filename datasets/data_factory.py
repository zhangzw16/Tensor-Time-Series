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
import glob
dataset_dir = '/home/ysc/workspace/Tensor-Time-Series/datasets/Tensor-Time-Series-Dataset/'

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
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'03',extra_para='PEMS03')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'04',extra_para='PEMS04')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'07',extra_para='PEMS07')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'08',extra_para='PEMS08')
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'20',extra_para='PEMS20')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'BAY',extra_para='PEMS_BAY')
    elif args.data == 'METRO':
        maker = Metro_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_SH',extra_para='SHMETRO')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_HZ',extra_para='HZMETRO')
    elif args.data == 'COVID':
        maker = COVID_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_CHI','COVID-CHI')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_US','COVID-US')
        # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_DEATHS','COVID-DEATHS')
    elif args.data == 'electricity':
        maker = electricity_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    elif args.data == 'weather':
        maker = weather_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    elif args.data == 'Jena_climate':
        maker = JenaClimate_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    elif args.data == 'shifts':
        maker = Shifts_data
        for set in ['dev_in','dev_out','eval_in','eval_out','train']:
            save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name+'_'+set,set)
    elif args.data == 'stocknet':
        maker = StockNet_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    elif args.data == 'nasdaq100':
        maker = Nasdaq100_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    elif args.data == 'crypto12':
        maker = Crypto_data
        save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    else:
        raise ValueError('Invalid dataset')
    # save_pickle(maker,os.path.join(dataset_dir,'Processed_Data'),data_name)
    return None
import re
def extract_date(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group(0) if match else ''
def Crypto_data():
    path = os.path.join(dataset_dir,'CrypTop12','price','raw')
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]
    arrays = []
    total_days = 1245
    unuseful = ['Date','SNo', 'Name', 'Symbol']
    # total_points = total_days*391
    absent = []
    for file_name in file_names:
        file_path = os.path.join(path,file_name)
        data = pd.read_csv(file_path)
        # data = pd.read_csv(file_path,sep = '\t',names = entries)
        data = data.drop(columns=unuseful)
        # print(data)
        if data.isna().any().any():
            data = data.fillna(0)
            print(data.isna().sum())
        # print(data)
        data = data.values
        if data.shape[0] < total_days:
            file_name = file_name.split('.')[0]+':'+str(data.shape[0])
            absent.append(file_name)
        else:
            arrays.append(data)
    data = np.stack(arrays, axis=1)
    print(absent)
    return data, data.shape, 'finance', data.shape, '1day'

def Nasdaq100_data():
    path = os.path.join(dataset_dir,'nasdaq100','nasdaq100','full')
    with open(os.path.join(path,'stock_name.txt')) as f:
        stock_names = [line.strip() for line in f]
    total_days = 191
    path = os.path.join(path,'stock_data_GOOGLE')
    entires = ['index','date','close','high','low','open','volume']
    excluded = []
    arrays = []
    for stock_name in stock_names:
        file_paths = glob.glob(os.path.join(path, stock_name+'*.csv'))
        if len(file_paths) == total_days:
            array = []
            file_paths = sorted(file_paths,key = lambda x: extract_date(x))
            for file_path in file_paths:
                data = pd.read_csv(file_path,names = entires)
                data = data.fillna('ffill')
                data = data.drop(columns=['index','date'])
                datain = data.values[1:]
                if datain.shape[0] < 391:
                    pad_length = 391 - datain.shape[0]
                    # 获取最后一个 5 维向量
                    last_vector = datain[-1]
                    # 创建一个填充数组，它的形状是 (pad_length, 5)，并且所有的元素都是最后一个 5 维向量
                    pad_array = np.tile(last_vector, (pad_length, 1))
                    # 将原始数组和填充数组沿着第 0 维堆叠起来
                    datain = np.vstack((datain, pad_array))
                if datain.shape[0] > 391:
                    datain = datain[:391]
                array.append(datain)
            array = np.vstack(array)
            arrays.append(array)
        else:
            excluded.append(stock_name)
    data = np.stack(arrays, axis=1)
    raw_shape = data.shape
    # data = np.expand_dims(data, axis=-1)
    data_type = 'finance'
    resolution = '1min'
    return data, data.shape, data_type, raw_shape, resolution
def StockNet_data():
    entries = ['date','open_price', 'high_price', 'low_price', 'close_price', 'adjust_close_price', 'volume']
    path = os.path.join(dataset_dir,'stocknet-dataset','price','raw')
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]
    arrays = []
    total_days = 1258
    total_points = total_days*391
    absent = []
    for file_name in file_names:
        file_path = os.path.join(path,file_name)
        data = pd.read_csv(file_path)
        # data = pd.read_csv(file_path,sep = '\t',names = entries)
        data = data.drop(columns=['Date'])
        # print(data)
        if data.isna().any().any():
            data = data.fillna(0)
            print(data.isna().sum())
        # print(data)
        data = data.values
        if data.shape[0] < total_days:
            file_name = file_name.split('.')[0]+':'+str(data.shape[0])
            absent.append(file_name)
        else:
            arrays.append(data)
    data = np.stack(arrays, axis=1)
    print(absent)
    return data, data.shape, 'finance', data.shape, '1day'
def weather_data():
    df = pd.read_csv(os.path.join(dataset_dir,'weather','weather.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    # df = df.resample('H').mean()
    df = df.fillna(method='ffill')
    print(df)
    data = df.values
    print(data)
    raw_shape = data.shape
    data = np.expand_dims(data, axis=-1)
    data_type = 'weather'
    resolution = '10mins'
    return data, data.shape, data_type, raw_shape, resolution
def JenaClimate_data():
    csv_path = os.path.join(dataset_dir,'Jena_climate','jena_climate_2009_2016.csv')
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Date Time'])
    if df.isna().any().any():
        df = df.fillna(0)
        print(df.isna().sum())
    data = df.values
    print(data)
    raw_shape = data.shape
    data = np.expand_dims(data, axis=-1)
    data_type = 'weather'
    resolution = '10mins'
    return data, data.shape, data_type, raw_shape, resolution


def Shifts_data(d_type):
    if d_type == 'dev_in':
        signal = os.path.join(dataset_dir,'Shifts','canonical-paritioned-dataset','shifts_dev_in.csv')
    elif d_type == 'dev_out':
        signal = os.path.join(dataset_dir,'Shifts','canonical-paritioned-dataset','shifts_dev_out.csv')
    elif d_type == 'eval_in':
        signal = os.path.join(dataset_dir,'Shifts','canonical-paritioned-dataset','shifts_eval_in.csv')
    elif d_type == 'eval_out':
        signal = os.path.join(dataset_dir,'Shifts','canonical-paritioned-dataset','shifts_eval_out.csv')
    elif d_type == 'train':
        signal = os.path.join(dataset_dir,'Shifts','canonical-paritioned-dataset','shifts_train.csv')
    else:
        raise ValueError('Invalid data type')
    df = pd.read_csv(signal)
    print(df.columns)
    if df.isna().any().any():
        df = df.fillna(0)
        print(df.isna().sum())
    # print(df)
    data = df.values
    # Ensure the time column is of datetime type
    raw_shape = data.shape
    data = np.expand_dims(data, axis=-1)
    data_type = 'traffic'
    data_resolution = '15mins'
    return data, data.shape, data_type, raw_shape, data_resolution
def COVID_data(d_type):
    if d_type == 'COVID-CHI':
        signal = pd.read_csv(os.path.join(dataset_dir,'COVID-CHI','data.csv'))
        print(signal)
        data = signal.values
        data = data[:,1:]
        raw_shape = data.shape
        data = np.reshape(data,(raw_shape[0],14,-1)).astype(float)
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
def electricity_data():
    df = pd.read_csv(os.path.join(dataset_dir,'electricity','electricity.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    # df = df.resample('H').mean()
    df = df.fillna(method='ffill')
    print(df)
    data = df.values
    print(data)
    raw_shape = data.shape
    data = np.expand_dims(data, axis=-1)
    data_type = 'energy'
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
        data = data.drop(data.columns[0],axis=1)
        data = data.drop(data.index[0])
        # print(data)
        data = data.values
        # print(data)
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
    parser.add_argument( '--data', type=str, default='stocknet', 
                        help='config file path')
    supported_datasets = ['Meter-LA',
                          'ETT_hour',
                          'JONAS_NYC',
                          'PEMS',
                          'METRO',
                          'COVID',
                          'electricity',
                          'weather',
                          'Jena_climate',
                          'shifts',
                          'stocknet',
                          'nasdaq100',
                          'crypto12']
    # 解析参数
    args = parser.parse_args()
    data_maker(args, 'train', )