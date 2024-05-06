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
    if threshold is None:
        return np.sqrt(np.mean(np.square(prediction - target)))
    else:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0][0]

def trunc_rmse(prediction, target, threshold=0):
    
    """
    Root Mean Square Error with Truncation (trunc_RMSE)
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target
            will be replaced by threshold in computing the s_rmse
    """
    predict_value = prediction.copy()
    target_value = target.copy()
    
    predict_value[predict_value<=threshold] = threshold
    target_value[target_value<=threshold] = threshold

    return np.sqrt(np.mean(np.square(predict_value - target_value)))
    
def mape(prediction, target, threshold=0):
    """
    Mean Absolute Percentage Error (MAPE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the mape.
    """
    assert threshold >= 0
    return (np.dot((np.abs(prediction - target) / (target + (1 - (target > threshold)))).reshape([1, -1]),
                   target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0, 0]


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

def trunc_mae(prediction, target, threshold=0):
    
    """
        Mean Absolute Error with Truncation (Trunc_MAE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target will be replaced in computing the mae
    """

    predict_value=prediction.copy()
    target_value = target.copy()

    predict_value[predict_value<=threshold] = threshold
    target_value[target_value<=threshold] = threshold

    return np.mean(np.abs(predict_value - target_value))

def smape(prediction, target, threshold=0):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the smape.
    """
    assert threshold >= 0

    predict_value = prediction[prediction >threshold]
    target_value = target[target >threshold]
    return np.mean(np.abs(predict_value - target_value) / ((np.abs(predict_value) + np.abs(target_value))*0.5))

def trunc_smape(prediction, target, threshold=0):
    """
    Symmetric Mean Absolute Percentage Error with Truncation (Trunc_SMAPE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be replaced in computing the trunc_smape.
    """
    predict_value = prediction.copy()
    target_value = target.copy()
    predict_value[predict_value<=threshold] = threshold
    target_value[target_value<=threshold] = threshold
    
    return np.mean(np.abs(predict_value - target_value) / ((np.abs(predict_value) + np.abs(target_value))*0.5))

'''
refer to DMSTGCN
'''

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds - labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds - labels) / labels
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)