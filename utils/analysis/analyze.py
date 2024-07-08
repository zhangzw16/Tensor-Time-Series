import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import csv

# Basic configuration
ModelType = ['MultiVar', 'Stat', 'Tensor_learned', 'Tensor_ng', 'Tensor_prior']
GraphInit = ['cosine', 'pearson', 'inverse_pearson', 'random']
Dataset = ['JONAS_NYC_bike', 'JONAS_NYC_taxi', 'Metr-LA','METRO_HZ', 'METRO_SH','PEMS03', 'PEMS07',
           'COVID_DEATHS', 'ETT_hour', 'electricity', 'weather', 'Jena_climate', 'nasdaq100']
ModelMap = {
    'Tensor':   ['NET3', 'DCRNN', 'GraphWaveNet', 'AGCRN', 'MTGNN', 'TTS_Norm', 'ST_Norm', 'GMRL'],
    'MultiVar': ['TimesNet', 'StemGNN', 'STGCN', 'AutoFormer', 'CrossFormer', 'PatchTST'],
    'Stat':     ['HM'],

}
His_len = [96]
Pred_len = 12

# Load data
METRIC_MAP = ['mae', 'mape', 'mse', 'rmse']
METRIC_ERROR = 'metric_error'
class YamlLoader:
    def __init__(self, path:str) -> None:
        self.yaml_path = path
        self.__load()

    def __load(self) -> dict:
        info = yaml.safe_load(open(self.yaml_path))
        self.info = info
        self.models = info['models']
        # self.datasets = info['datasets']
        self.model_type = info['model_type']
        # self.his_len, self._pred_len = info['his_pred']
        self.his_len = info['his_len']
        self.pred_len = info['pred_len']
        self.data_mode = info['data_mode']
        # print(self.models, self.datasets)

    def get_metrics(self, model_name, dataset_name):
        # print(model_name, self.models)
        # print(dataset_name, self.datasets)
        # if model_name in self.models and dataset_name in self.datasets:
        if model_name in self.models and dataset_name in self.info:
            metrics =  self.info[dataset_name][model_name]
            if isinstance(metrics, str):
                metrics = METRIC_ERROR
            return metrics
        else:
            return None
    def get_data_mode(self):
        return self.data_mode
        
class Collector:
    def __init__(self, root_path:str):
        self.root_path = root_path
        pass

    def collect(self, his:int, pred:int):
        yaml_base = os.path.join(self.root_path, f'{his}-{pred}', 'log')
        yaml_files = os.listdir(yaml_base)
        res = {dataset:{} for dataset in Dataset}
        for yaml_file in yaml_files:
            yaml_path = os.path.join(yaml_base, yaml_file)
            # if '_2_' not in yaml_path:
            #     continue
            loader = YamlLoader(yaml_path)
            for dataset in Dataset:
                for key in ModelMap:
                    model_list = ModelMap[key]
                    for model_name in model_list:
                        metrics = loader.get_metrics(model_name, dataset)
                        data_mode = loader.get_data_mode()
                        # print(metrics)
                        # print(f'{dataset} {model_name} {metrics}')
                        if metrics is not None:
                            if model_name not in res[dataset]:
                                # res[dataset][model_name] = {}
                                # res[dataset][model_name]['metrics'] = metrics
                                # res[dataset][model_name]['data_mode'] = data_mode
                                res[dataset][f'{model_name}-{data_mode}'] = metrics
        # print(res)
        return res
    
    def collect_all(self, output_path:str, output:str='csv'):
        res = {}
        for his in His_len:
            for pred in [Pred_len]:
                res[f'{his}-{pred}'] = self.collect(his, pred)
        if output == 'csv':
            self.to_csv(output_path, res)

    def to_csv(self, output_path, res:dict):
        for his_pred in res:
            with open(os.path.join(output_path, f'{his_pred}-merge-v2.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['his_pred', 'dataset', 'model', 'model_type','mae', 'mape', 'rmse', 'smape'])
                for dataset in res[his_pred]:
                    for model in res[his_pred][dataset]:
                        # f.write(f'{his_pred},{dataset},{model}')
                        model_type = 'Tensor' if model in ModelMap['Tensor'] else 'MultiVar' if model in ModelMap['MultiVar'] else 'Stat'
                        # data_mode = res[his_pred][dataset][model]['data_mode']
                        if res[his_pred][dataset][model] == METRIC_ERROR:
                            writer.writerow([his_pred, dataset, f'{model}', METRIC_ERROR, METRIC_ERROR, METRIC_ERROR, METRIC_ERROR])
                            continue
                        mae = res[his_pred][dataset][model]['mae']
                        mape = res[his_pred][dataset][model]['mape']
                        rmse = res[his_pred][dataset][model]['rmse']
                        smape = res[his_pred][dataset][model]['smape']
                        writer.writerow([his_pred, dataset, f'{model}', model_type, mae, mape, rmse, smape])
                # f.flush()

if __name__ == '__main__':
    collector = Collector('/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/output')
    collector.collect_all('/home/zhuangjiaxin/workspace/TensorTSL/Tensor-Time-Series/output/csv')
    
