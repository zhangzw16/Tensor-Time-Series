import pandas as pd
import csv
import os

class ResItem:
    def __init__(self, model, dataset, his_pred, mae, mape, rmse, smape):
        self.model = model
        self.dataset = dataset
        self.his_pred = his_pred
        self.metric = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'smape': smape
        }


def read_results(res_path:str, dataset_list:list):
    results = {}
    df = pd.read_csv(res_path)
    for index, row in df.iterrows():
        model = row['model']
        dataset = row['dataset']
        his_pred = row['his_pred']
        mae = row['mae']
        mape = row['mape']
        rmse = row['rmse']
        smape = row['smape']
        if dataset in dataset_list:
            if model not in results:
                results[model] = {}
            results[model][dataset] = ResItem(model, dataset, his_pred, mae, mape, rmse, smape)
    return results

def read_dataset(dataset_path:str):
    dataset = {}
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        dataset_name = row['Dataset']
        time_points = row['TimePTs']
        nodes = row['Nodes']
        features = row['Features']
        freq = row['Freq']
        train_mean = row['TrainMean']
        train_std = row['TrainStd']
        test_mean = row['TestMean']
        test_std = row['TestStd']
        node_corr_mean = row['NodeCorrMean']
        node_corr_std = row['NodeCorrStd']
        feature_corr_mean = row['FeatureCorrMean']
        feature_corr_std = row['FeatureCorrStd']

        dataset_mean_dis = test_mean / train_mean -1
        dataset_std_dis = test_std / train_std -1
        
        if str(features) == '1':
            dataset_type = 'MTS'
        else:
            dataset_type = 'TS'
        if dataset_name not in dataset:
            dataset[dataset_name] = {}
        dataset[dataset_name] = {
            'name': dataset_name,
            'type': dataset_type,
            'time_points': time_points,
            'nodes': nodes,
            'features': features,
            'freq': freq,
            'node_corr_mean': node_corr_mean,
            'node_corr_std': node_corr_std,
            'feature_corr_mean': feature_corr_mean,
            'feature_corr_std': feature_corr_std,
            'dataset_mean_dis': dataset_mean_dis,
            'dataset_std_dis': dataset_std_dis
        }
    return dataset



