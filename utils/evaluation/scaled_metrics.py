import numpy as np

def mase(hist, pred, truth, season_length=1, threshold=None):
    """
    Implement (seasonal) MASE proposed by Hyndman and Koehler (2005).
    Ref paper: Another look at measures of forecast accuracy

    Args:
    hist: history with size [batch, time, ...]
    pred: prediction with size [batch, time, ...]
    truth: observation with size [batch, time, ...]
    """
    hist_len = hist.shape[1]
    denominator = np.abs(hist[:,season_length:, :] - hist[:, :-season_length, :])
    mase = np.mean(np.abs(pred - truth)) / (np.mean(denominator))

    return mase

def rmsse(hist, pred, truth, season_length=1, threshold=None):
    """
    Implement (seasonal) RMSSE

    Args:
    hist: history with size [batch, time, ...]
    pred: prediction with size [batch, time, ...]
    truth: observation with size [batch, time, ...] 
    """
    hist_len = hist.shape[1]
    denominator = np.square(hist[:,season_length:, :] - hist[:, :-season_length, :])
    rmsse = np.sqrt(np.mean(np.square(pred - truth))) / np.sqrt(np.mean(denominator))

    return rmsse