import torch
from torch.utils.data import DataLoader
from models.model_base import ModelBase

'''
Only for cuda
'''
class AutoBatch:
    def __init__(self, model:ModelBase, dataset, initial_batch:int=512, strategy:str='exp', verbose:bool=True):
        self.dataset = dataset
        self.model = model
        self.init_batch = initial_batch
        self.max_batch = 0
        self.strategy = strategy
        self.verbose = verbose
        self.update_func_map = {
            'exp': self.exp_strategy,
            'linear': self.linear_strategy,
        }
        self.update_func = self.update_func_map[self.strategy]
        self.device_num = torch.cuda.device_count()

    def get_total_memory(self):
        total_mem_B = torch.cuda.get_device_properties(0).total_memory
        total_men_KB = total_mem_B / 1024
        total_mem_MB = total_men_KB / 1024
        total_mem_GB = total_mem_MB / 1024
        return total_mem_GB

    def is_out_of_memory(self, exception):
        return (
            isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory" in exception.args[0]
        )

    def load_to_GPU(self, batch):
        self.model.set_device('cuda')
        dataloader = DataLoader(self.dataset, batch_size=batch, shuffle=False, drop_last=False)
        for seq in dataloader:
            seq = seq.to('cuda')
            pred, truth = self.model.forward(seq)
            break
        torch.cuda.empty_cache()

    def search_batch(self):
        while self.init_batch > 1:
            try:
                self.load_to_GPU(self.init_batch)
                if self.verbose:
                    print(f"Batch: {self.init_batch} OK!")
                return self.init_batch
            except RuntimeError as exception:
                if self.init_batch > 1 and self.is_out_of_memory(exception):
                    print(f"Batch: {self.init_batch} failed...")
                    self.update_func()
                    print(f"Try next batch size: {self.init_batch}")
                else:
                    raise

    def exp_strategy(self, factor=2):
        self.init_batch = max(self.init_batch // factor, 1)

    def linear_strategy(self, factor=16):
        self.init_batch = max(self.init_batch - factor, 1)