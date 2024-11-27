import numpy as np
import torch

'''
Masked MAE
'''
def masked_mae(preds, labels, threshold=0.0):
    if np.isnan(threshold):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != threshold)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)