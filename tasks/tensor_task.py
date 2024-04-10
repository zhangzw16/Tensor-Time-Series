import os
import torch
import numpy as np
from tasks.task_base import TaskBase
from models.model_base import TensorModelBase

class TensorTask(TaskBase):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.model = TensorModelBase()
        self.mode = 'train'
        self.model.set_device('cuda')

    def get_data(self):
        return super().get_data()

    def train(self):
        for i in range(self.max_epoch):
            self.model.train()
            for () in self.train_loader:
                self.model.forward()
                print('TODO:...')
        # log information
        # ...
    
    def early_stop(self):
        return super().early_stop()

    def valid(self):
        return super().valid()

    def test(self):
        return super().test()