import numpy as np
from torch.utils.data import Dataset


class TensorTSDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        his_len: int,
        pred_len: int,
    ):
        self.data = data
        self.his_len = his_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.his_len - self.pred_len

    def __getitem__(self, idx):
        h_start = idx
        h_end = idx + self.his_len
        p_start = h_end
        p_end = h_end + self.pred_len
        his_data = self.data[h_start:h_end]
        pred_data = self.data[p_start:p_end]
        seq_data = np.concatenate([his_data, pred_data], axis=0, dtype=np.float32)
        # idx = np.array([idx])
        return seq_data, idx
