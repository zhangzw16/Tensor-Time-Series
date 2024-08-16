import numpy as np

'''
refer to UCTB: https://github.com/uctb/UCTB
'''

def rmse(prediction, target, threshold=None):
    """
    Root Mean Square Error (RMSE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target will be removed in computing the rmse
    """
    # print(prediction.shape)
    if threshold is None:
        return np.sqrt(np.mean(np.square(prediction - target)))
        # return (np.mean((prediction-target)**2, axis=(0,2))**0.5)
    else:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0][0]
    
def mape(prediction, target, threshold=0):
    """
    Mean Absolute Percentage Error (MAPE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the mape.
    """
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    
    # Avoid division by zero
    mask = target != 0
    return np.mean(np.abs((prediction[mask] - target[mask]) / target[mask]))


def mae(prediction, target, threshold=None):
    """
    Mean Absolute Error (MAE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target will be removed in computing the mae
    """
    if threshold is None:
        return np.mean(np.abs(prediction - target))
    else:
        return (np.dot(np.abs(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0, 0]

def smape(prediction, target, threshold=0):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the smape.
    """
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    denominator = (np.abs(target) + np.abs(prediction)) / 2.0
    diff = np.abs(target - prediction) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def pcc(pred, labels, threshold=None):
    pred = pred.reshape(-1)
    labels = labels.reshape(-1)
    pcc = np.corrcoef(pred, labels)[0, 1]
    return pcc

def r_square(pred, labels, threshold=None):
    """
    R^2 score

    Args:
        pred(ndarray): prediction with shape [batch_size, time, ...]
        labels(ndarray): same shape with prediction, [batch_size, time, ...]
        threshold(float): data smaller or equal to threshold in target will be removed in computing the r2
    """
    pred_len = pred.shape[1]
    pred = pred.reshape(pred.shape[0], pred_len, -1)
    labels = labels.reshape(labels.shape[0], pred_len, -1)
    mean = np.expand_dims(np.mean(labels, axis=1), axis=1) # mean of truth over time
    ss_tot = np.sum((labels - mean) ** 2)
    ss_res = np.sum((labels - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2