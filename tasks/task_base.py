import os
import sys
import math
import torch

class TaskBase:
    def __init__(self, configs:dict={}) -> None:
        self.configs = configs
        self.best_valid_loss =  math.inf
        self.early_stop_max = configs['early_stop_max']
        self.early_stop_cnt = 0

    def ensure_output_dir(self, path:str):
        if not os.path.exists(path):
            os.makedirs(path,)

    def early_stop(self, valid_loss, epoch_info:dict={}, save_dir:str='')->bool:
        # check loss and update
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.early_stop_cnt = 0
            if self.model is not None:
                if save_dir == '':
                    save_dir = self.output_dir
                save_path = os.path.join(save_dir, 'model.pth')
                self.model.save_model(save_path)
                print(f'model saved in: {save_path}')
                self.best_epoch_info = epoch_info
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

    def test_model_io_shape(self):
        pass