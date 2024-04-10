
import os
import torch

class TaskBase:
    def __init__(self, configs) -> None:
        self.configs = configs
        self.device = configs['device']
    
    def ensure_work_dir(self):
        raise NotImplemented
    
    def get_data(self):
        raise NotImplemented
    
    def train(self):
        pass

    def valid(self):
        pass

    def test(self):
        pass

    def early_stop(self):
        pass
        