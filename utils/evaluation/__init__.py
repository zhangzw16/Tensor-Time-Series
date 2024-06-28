import os
import importlib
import numpy as np

class Evaluator:
    def __init__(self, metric_list:list, thres_dict:dict={}) -> None:
        self.load_metrics(metric_list, thres_dict)


    def load_metrics(self, metric_list:list, thres_dict):
        metric_pkg_name = f"{__name__}.metrics"
        self.metric_pkg = importlib.import_module(metric_pkg_name)
        self.metrics_map = {}
        self.metrics_thres_map = thres_dict
        attr_list = dir(self.metric_pkg)
        for metric in metric_list:
            if metric in attr_list:
                self.metrics_map[metric] = getattr(self.metric_pkg, metric)

    def eval(self, pred, truth, verbose=False):
        result = {}
        for metric_name in self.metrics_map:
            metric_func = self.metrics_map[metric_name]
            if metric_name in self.metrics_thres_map:
                thres = float(self.metrics_thres_map[metric_name])
                metric_result = metric_func(pred, truth, thres)
            else:
                metric_result = metric_func(pred, truth)
            result[metric_name] = metric_result
        if verbose:
            self.print_result(result)
        return result
    
    def print_result(self, result:dict):
        print('----- show results -----')
        for name in result:
            print(f'{name}: {result[name]:.3f}')
        print('--------- end ----------')