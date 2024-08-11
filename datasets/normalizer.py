import torch
import numpy as np

class IdentityNormalizer:
    def __init__(self) -> None:
        pass
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

class StandNormalizer:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def transform(self, data):
        data_norm = (data-self.mean) / (self.std)
        return data_norm
    
    def inverse_transform(self, data):
        invser_data = data * self.std + self.mean
        return invser_data