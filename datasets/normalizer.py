import torch
import numpy as np
from sklearn.preprocessing import StandardScaler



class DoNothing:
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
    
class SKlearnNormalizer:
    def __init__(self, data) -> None:
        self.scaler = StandardScaler()
        raw_shape = data.shape
        self.feature_num = raw_shape[1]*raw_shape[2]
        data = self.reshape(data,(data.shape[0], -1))
        self.scaler.fit(data)
        data = self.reshape(data,raw_shape)
    def reshape(self,data,shape):
        try :
            data = data.view(shape)
        except:
            data = data.reshape(shape)
        return data

    def transform(self, data):
        try:
            device = data.device
        except:
            device = 'cpu'
        dtype = data.dtype
        if device == 'cpu':
            raw_shape = data.shape
            data = self.reshape(data,(-1, self.feature_num))
            data_norm = self.scaler.transform(data)
            data = torch.tensor(self.reshape(data_norm,raw_shape))
        else:
            raw_shape = data.shape
            data = self.reshape(data,(-1, self.feature_num))
            data_norm = self.scaler.transform(data.cpu())
            data = torch.tensor(self.reshape(data_norm,raw_shape)).float()
            data = data.to(device)
        return data
    
    def inverse_transform(self, data):
        device = data.device
        raw_shape = data.shape
        if device == 'cpu':
            data = self.reshape(data,(-1, self.feature_num))
            inverse_data = self.scaler.inverse_transform(data)
            data = torch.tensor(self.reshape(inverse_data,raw_shape))
        else:
            data = self.reshape(data,(-1, self.feature_num))
            inverse_data = self.scaler.inverse_transform(data.cpu().detach().numpy())
            data = torch.tensor(self.reshape(inverse_data,raw_shape)).float()
            data = data.to(device)
        return data