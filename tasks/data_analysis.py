import numpy as np
import pickle as pkl
import os
from scipy.spatial import distance
from scipy.fftpack import fft 
from scipy import stats
import math
from itertools import permutations
from scipy.special import factorial
from collections import Counter

def calculate_correlation(matrix1, matrix2):
    # 将矩阵转换为一维数组
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()

    # 计算相关性
    correlation = np.corrcoef(matrix1, matrix2)

    return correlation[0, 1]
def calculate_cosine_distance(matrix1, matrix2):
# 将矩阵转换为一维数组
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()

    # 计算余弦距离
    cosine_distance = distance.cosine(matrix1, matrix2)

    return cosine_distance
def calculate_euclidean_distance(matrix1, matrix2):
    # 将矩阵转换为一维数组
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()

    # 计算欧式距离
    euclidean_distance = distance.euclidean(matrix1, matrix2)

    return euclidean_distance

def calculate_manhattan_distance(matrix1, matrix2):
    # 将矩阵转换为一维数组
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()

    # 计算曼哈顿距离
    manhattan_distance = distance.cityblock(matrix1, matrix2)

    return manhattan_distance
def calculate_minkowski_distance(self,matrix1, matrix2):
    # 将矩阵转换为一维数组
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()

    # 计算明可夫斯基距离
    minkowski_distance = distance.minkowski(matrix1, matrix2, 3)

    return minkowski_distance

def find_top_k_periods(ts, k):
    # 计算FFT
    if np.nonzero(ts)[0].shape[0] == 0:
        return np.ones(k), np.ones(k)
    fft = np.abs(np.fft.fft(ts))
    frequencies = np.fft.fftfreq(len(ts))

    # 找到最大的k个频率
    indices = np.argsort(np.abs(fft[1:-1]))[-k:]
    periods = 1 / (frequencies[indices]+1e-10)
    strength = fft[indices]/np.sum(fft)
    return np.abs(periods.astype(int)), strength

def find_top_k_periods_acf(ts,k):
    if np.nonzero(ts)[0].shape[0] == 0:
        return np.ones(k), np.ones(k)
    acf = np.correlate(ts,ts,mode = 'full')
    acf = acf[acf.size//2:]
    indices = np.argsort(np.abs(acf))[-k:]
    periods = indices
    return np.abs(periods.astype(int))



def sliding_window(sequence, embedding_dim = 10, time_delay = 1):
    # 生成滑动窗口
    sequence = np.array(sequence)
    k = sequence.shape[0]-(embedding_dim- 1)*time_delay
    if k <= 0:
        raise ValueError("The sequence is too short to generate a sliding window.")
    windows = []
    k = sequence.shape[0]-(embedding_dim- 1)*time_delay
    for i in range(k):
        window = sequence[i:i+(embedding_dim-1)*time_delay+1:time_delay]
        windows.append(window)
    return np.array(windows)

def permutation_entropy(windows):
    # 创建所有可能的排列
    num_windows = windows.shape[0]
    embedding_dim = windows.shape[1]
    perms = list(permutations(range(embedding_dim)))
    mapdict = {p:i for i, p in enumerate(perms)}
    inv_mapdict = {v: k for k, v in mapdict.items()}

    # 计算每个排列在时间序列中出现的次数
    counts = Counter([tuple(np.argsort(windows[i])) for i in range(num_windows)])

    # 计算每个排列的概率
    probs = np.array([counts[inv_mapdict[i]] for i in range(len(perms))]) / (num_windows)

    # 计算置换熵
    probs = probs[probs > 0]
    pe = -np.nansum(probs * np.log2(probs))

    return pe

def core_function(matrix):
    return np.linalg.norm(matrix,1)

def tensor_to_value(windows, core = core_function):
    # 将滑动窗口转换为分布
    num_windows = windows.shape[0]
    embedding_dim = windows.shape[1]
    orders = []
    for i in range(num_windows):
        window = windows[i]
        quantify = np.zeros(embedding_dim)
        for j in range(embedding_dim):
            quantify[j] = core(window[j])
        orders.append(quantify)
    return np.array(orders)
    



        


class DataAnalysis:
    def __init__(self, pkl_path):
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        self.data = self.data_pkl['data']
        self.data_shape = self.data_pkl['data_shape']
        self.raw_shape = self.data_pkl['raw_shape']
        self.data_type = self.data_pkl['data_type']
        self.process = {'corelation': calculate_correlation,
                        'cosine_distance': calculate_cosine_distance,
                        'euclidean_distance': calculate_euclidean_distance,
                        'manhattan_distance': calculate_manhattan_distance,
                        'minkowski_distance': calculate_minkowski_distance
                        }
    
    def eval_tensor(self,axis = 1, method = 'corelation'):
        num = self.data.shape[axis]
        matrices = np.split(self.data, num, axis=axis)
        # matrices = [matrix.reshape(t*m) for matrix in matrices]
        distances = np.zeros((num, num))
        for i in range(num):
            for j in range(i, num):
                distances[i, j] = self.process[method](matrices[i], matrices[j])
                distances[j, i] = distances[i, j]
        return distances

    def find_periods_byacf(self, k = 3):

        # 对每个元素分别计算其最可能的周期
        merge_feature = self.data.reshape(self.data.shape[0], -1)
        res = np.array([find_top_k_periods_acf(merge_feature[:, i], k) for i in range(merge_feature.shape[1])])
        periods = res
        top_k_periods = []
        for i in range(k):
            period = periods[:,i].flatten()
            # mean = np.mean(period)
            # sigma = np.std(period)
            # outliers = np.where(abs(period - mean)> 2*sigma)
            # np.delete(period, outliers)
            mode = stats.mode(period)[0]
            var  = np.var(periods[:,i])
            ratio = np.sum(periods[:,i] == mode) / periods.shape[0]
            # strength_p = strength[periods[:,i] == mode]
            # avg_strength = np.mean(strength_p)
            dict = {'period': mode, 'ratio_across_variants': ratio, 'varience': var}
            top_k_periods.append(dict)

        #for large to small
        top_k_periods = top_k_periods[::-1]
        return top_k_periods

    def find_periods(self, k = 3):

        # 对每个元素分别计算其最可能的周期
        merge_feature = self.data.reshape(self.data.shape[0], -1)
        res = np.array([find_top_k_periods(merge_feature[:, i], k) for i in range(merge_feature.shape[1])])
        periods = res[:, 0, :]
        strength = res[:, 1, :]
        
        top_k_periods = []
        for i in range(k):
            period = periods[:,i].flatten()
            mean = np.mean(period)
            sigma = np.std(period)
            outliers = np.where(abs(period - mean)> 2*sigma)
            np.delete(period, outliers)
            mode = stats.mode(period)[0]
            var  = np.var(period)
            ratio = np.sum(period == mode) / period.shape[0]
            strength_p = strength[period == mode]
            avg_strength = np.mean(strength_p)
            dict = {'period': mode, 'ratio_across_variants': ratio, 'varience': var,'energy_density': avg_strength}
            top_k_periods.append(dict)

        #for large to small
        top_k_periods = top_k_periods[::-1]
        return top_k_periods
    def tensor_permutation_entropy(self ,embedding = 4, time_delay = 1):
        sequence = self.data
        windows = sliding_window(sequence, embedding_dim = embedding, time_delay = time_delay)
        windows = tensor_to_value(windows)
        pe = permutation_entropy(windows)
        return pe



if __name__ == "__main__":
    pkl_path = '/home/ysc/workspace/Tensor-Time-Series/datasets/Tensor-Time-Series-Dataset/Processed_Data/Metr-LA/Metr-LA.pkl'
    # pkl_path = "/home/ysc/workspace/Tensor-Time-Series/datasets/Tensor-Time-Series-Dataset/Processed_Data/JONAS_NYC_bike/JONAS_NYC_bike.pkl"
    data_analysis = DataAnalysis(pkl_path)
    # periods = data_analysis.find_periods_byacf(k = 3)
    corr = data_analysis.eval_tensor(axis = 2, method = 'corelation')
    # pe = data_analysis.tensor_permutation_entropy(embedding = 48, time_delay = 7)
    # sequence=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # sequence = np.linspace(0,479,480)
    # sequence = np.random.permutation(sequence)
    # sequence = np.reshape(sequence,(10,4,3,4))
    # # sequence = np.array(sequence)
    # sequence = tensor_to_value(sequence)
    # en = permutation_entropy(sequence)

    # windows = sliding_window(sequence, embedding_dim = 3, time_delay = 2)
    # print(windows)
    # # evals = data_analysis.find_periods(k = 10)
    # # evels = np.where(evals > 0)[0]
    # # evals = np.mean(evals[evels])
    # matrix1 = data_analysis.data[:,:,0]
    # matrix2 = data_analysis.data[:,:,1]
    # correlation = data_analysis.calculate_correlation(matrix1, matrix2)
    # print(f"Correlation between matrix1 and matrix2: {correlation}")