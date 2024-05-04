
import os
import math
import torch

class TaskBase:
    def __init__(self, configs:dict={}) -> None:
        self.configs = configs
        self.best_valid_loss =  math.inf
        self.early_stop_max = configs['early_stop_max']
        self.early_stop_cnt = 0
    def ensure_work_dir(self, path:str):
        raise NotImplemented

    def early_stop(self, valid_loss)->bool:
        # check loss and update
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.early_stop_cnt = 0
        else:
            self.early_stop_cnt += 1
        # check cnt and return stop flag
        if self.early_stop_cnt > self.early_stop_max:
            return True
        else:
            return False
        
    def train(self):
        pass

    def test(self):
        pass