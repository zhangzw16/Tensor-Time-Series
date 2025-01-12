from tasks.data_analysis import DataAnalysis
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import os
#set dir to the path of this file
# from tsfeature import tsfeature
from tsfeatures.tsfeatures.tsfeatures import tsfeatures
import pandas as pd

root_path = '/home/ysc/workspace/Tensor-Time-Series/datasets/data/'
dataset_freq = {
        'JONAS_NYC_bike': '30T', 'JONAS_NYC_taxi': '30T', 'Metr-LA': '5T', 'METRO_HZ': '15T', 
        'METRO_SH': '15T', 'PEMSBAY': '5T', 
        'COVID_CHI': '1H', 'COVID_US': '1H',
        'COVID_DEATHS': '1D', 'ETT_hour': '1H', 'electricity': '1H',
        'weather': '15T', 'Jena_climate': '10T', 
        'nasdaq100': '1T', 
        'stocknet': '1H', 'crypto12': '1H'}
Datasets =  ['JONAS_NYC_bike', 'JONAS_NYC_taxi', 'Metr-LA','METRO_HZ', 'METRO_SH',#'PEMS03', 'PEMS07', 'PEMS20', 
             'COVID_CHI', 'COVID_US',
                'COVID_DEATHS',
                'ETT_hour','electricity',
                'weather', 'Jena_climate',
                'nasdaq100', 'stocknet','crypto12']
FREQS = {'1T':24*60,'5T':24*12,'10T':24*6,'15T': 24*4,'30T': 24*2,
         '1H': 24,'2H': 12,'1D': 1,
         'H': 24, 'D': 1,
         'M': 12, 'Q': 4,
         'W':1, 'Y': 1}

# pkl_path = root_path + 'weather/weather.pkl'
dataset_res = {}
# pkl_path = '/home/ysc/workspace/Tensor-Time-Series/datasets/Tensor-Time-Series-Dataset/Processed_Data/Metr-LA/Metr-LA.pkl'
from tqdm import tqdm

class tsfDataAnalysis():
    def __init__(self, pkl_path, read_features = True, feature_read_path = '/nas/datasets/ysc/Tensor-Time-Series/output/tsfeatures'):
        self.pkl_path = pkl_path
        self.dataset = pkl_path.split('/')[-2]
        self.data = self.load_data()
        self.ts = self.data['data']
        self.data_shape = self.data['data_shape']
        self.start_timestamp = '2023-01-01 00:00:00'
        # self.interval_map = {'15mins': '15T', '1hour': '1H', '1day': '1D'}
        # self.interval = self.interval_map[self.data['temporal_resolution']]
        self.interval = dataset_freq[self.dataset]
        if read_features:
            if os.path.exists(os.path.join(feature_read_path,self.dataset+'.csv')):
                self.tsfdf = pd.read_csv(os.path.join(feature_read_path,self.dataset+'.csv'))
            else:
                print('No features found, calculating...')
                # self.tsfdf = self.get_all_features(feature_read_path)
                # self.tsfdf = pd.read_csv(os.path.join(feature_read_path,self.dataset+'.csv'))
        # self.tsfdf = self.reshape_timeseries()


    def load_data(self):
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    def check_and_add_key(self, key, value):
        if key not in self.data:
            self.data[key] = value
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Key '{key}' added to the pkl file.")
        else:
            print(f"Key '{key}' already exists in the pkl file.")

    def get_row_as_dict(self, df, unique_id_value):
        # 找到 unique_id 列等于 unique_id_value 的行
        row = df[df['unique_id'] == unique_id_value]
        
        if row.empty:
            return None
        
        # 将这一行不为空的值及其对应的列名以字典的形式返回
        row_dict = row.dropna(axis=1).to_dict(orient='records')[0]
        for key, value in row_dict.items():
            if isinstance(value, float):
                row_dict[key] = round(value, 3)
        return row_dict
    
    def plot_single_ts(self, N = 0, M = 0, start = 0, end = None):
        if end is None or end > self.ts.shape[0]:
            series = self.ts[start:, N, M]
            x = np.arange(start, start + len(series))
        else:
            series = self.ts[start:end, N, M]
            x = np.arange(start, end)
        return x, series
    
    def plot_all_modality(self, N = 0, M = 0, start = 0, end = None):
        if end is None or end > self.ts.shape[0]:
            series = self.ts[start:, N, :]
            x = np.arange(start, start + len(series))
        else:
            series = self.ts[start:end, N, :]
            x = np.arange(start, end)
        return x, series
            
    def avaliable_features(self,N = 0,M = 0):
        features = self.get_row_as_dict(self.tsfdf, f'N_{N}_M_{M}')
        return features
    

        
    def get_single_ts(self, node, feature):
        unique_id = f'N_{node}_M_{feature}'
        series = self.ts[:, node, feature]
        id_series = [unique_id] * len(series)
        ds = pd.date_range(start=self.start_timestamp, periods=len(series), freq=self.interval)
        df = pd.DataFrame({'unique_id': id_series, 'ds': ds, 'y': series})
        return df
    
    def get_all_features(self):
        outpath = self.feature_read_path
        time_steps, nodes, features = self.data_shape
        # nodes = 2
        # features = 2
        all_features = []
        for node in tqdm(range(nodes)):
            for feature in tqdm(range(features)):
                df = self.get_single_ts(node, feature)
                TSfeatures = tsfeatures(df,freq = FREQS[self.interval])
                all_features.append(TSfeatures)
        df_all = pd.concat(all_features)
        df_all.to_csv(os.path.join(outpath,self.dataset+'.csv'), index=False)
        df_avgfeatures = df_all.mean(numeric_only=True).to_frame().T
        # columns = df_all.columns
        df_avgfeatures['unique_id'] = self.dataset
        df_avgfeatures['N'] = self.data_shape[1]
        df_avgfeatures['M'] = self.data_shape[2]
        df_avgfeatures.to_csv(os.path.join(outpath,self.dataset+'_avg.csv'), index=False)
        return df_avgfeatures
    
    def aggregate_features(self):
        import glob
        all_files = glob.glob('/nas/datasets/ysc/Tensor-Time-Series/output/tsfeatures/*_avg.csv')
        df_all = pd.concat([pd.read_csv(file) for file in all_files])
        df_all.to_csv('/nas/datasets/ysc/Tensor-Time-Series/output/tsfeatures/all_datasets.csv', index=False)
        return df_all
        
    def reshape_timeseries(self):
        time_steps, nodes, features = self.data_shape
        reshaped_data = []

        for node in tqdm(range(nodes)):
            for feature in tqdm(range(features)):
                for time_step in range(time_steps):
                    unique_id = f'N_{node}_M_{feature}'
                    series = self.ts[time_step, node, feature]
                    id_series = [unique_id] * len(series)
                    ds = pd.date_range(start=self.start_timestamp, periods=len(series), freq=self.interval)
                    df = pd.DataFrame({'unique_id': id_series, 'ds': ds, 'y': series})
                    reshaped_data.append
        df = pd.DataFrame(reshaped_data, columns=['unique_id', 'ds', 'y'])
        return df

    # def find_periods(self, k = 1):
    #     data = self.data
    #     data_shape = data.shape
    #     period = []
    #     for i in range(data_shape[0]):
    #         for j in range(data_shape[1]):
    #             if data[i][j] == 0:
    #                 continue
    #             if j + k < data_shape[1] and data[i][j + k] == 0:
    #                 period.append(j)
    #     return period

if __name__ == '__main__':
    # for dataset in Datasets:
    dataset = 'METRO_HZ'
    pkl_path = root_path + dataset + '/' + dataset + '.pkl'
    data_analysis = tsfDataAnalysis(pkl_path)
    data_analysis.aggregate_features()
    # data_analysis = tsfDataAnalysis(pkl_path)
    # print(data_analysis.data_shape)
    # print(data_analysis.data['temporal_resolution'])
    # data_analysis.get_all_features('/home/ysc/workspace/Tensor-Time-Series/output/tsfeatures')
    # all_data_features = []
    # for dataset in tqdm(dataset_freq.keys()):
    #     pkl_path = root_path + dataset + '/' + dataset + '.pkl'
    #     data_analysis = tsfDataAnalysis(pkl_path)
    #     print(data_analysis.dataset)
    #     print(data_analysis.data_shape)
    #     print(data_analysis.interval)
    #     avg = data_analysis.get_all_features('/home/ysc/workspace/Tensor-Time-Series/output/tsfeatures')
    #     all_data_features.append(avg)
    # df_all = pd.concat(all_data_features)
    # df_all.to_csv('/home/ysc/workspace/Tensor-Time-Series/output/tsfeatures/all_datasets.csv', index=False)
    # df = data_analysis.get_single_ts(0, 0)
    # features = tsfeatures(df,freq = FREQS[data_analysis.interval])
    # print(df)
    # dataset_res[dataset] = data_analysis.data_shape
    
    # data_analysis = tsfDataAnalysis(pkl_path)
    # print(data_analysis.data_shape)
   