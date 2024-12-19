import os
import pickle as pkl
import numpy as np
import random

# from .normalizer import Sklearn_StandNormalizer, DoNothing, StandNormalizer

os.chdir(os.path.dirname(__file__))

class MoveSample(object):
    def __init__(self, feature_step, feature_stride, feature_length, target_length):
        self.feature_step = feature_step
        self.feature_stride = feature_stride
        self.feature_length = feature_length
        self.target_length = target_length

    def general_move_sample(self, data):
        feature = []
        target = []
        # sample_num = len(data) - window_size + 1
        # window_size = feature_length + (feature_step-1)*feature_stride + target_length
        for i in range(len(data) - self.feature_length -
                       (self.feature_step-1)*self.feature_stride - self.target_length + 1):
            feature.append([data[i + step*self.feature_stride: i + step*self.feature_stride + self.feature_length]
                            for step in range(self.feature_step)])
            target.append(data[i + (self.feature_step-1) * self.feature_stride + self.feature_length:\
                               i + (self.feature_step-1) * self.feature_stride + self.feature_length + self.target_length])

        return np.array(feature), np.array(target)

class ST_MoveSample(object):
    '''
    This class can converts raw data into temporal features including closenss, period and trend features.

    Raw data format: [time, dim1, dim2]

    output format: X: [window_num, dim1, dim2, (window_size = closeness_len + period_len + trend_len)]
                Y: [window_num, dim1, dim2, target_length]

    Args:
        closeness_len(int):The length of closeness data history. The former consecutive ``closeness_len`` time slots
            of data will be used as closeness history.
        period_len(int):The length of period data history. The data of exact same time slots in former consecutive
            ``period_len`` days will be used as period history.
        trend_len(int):The length of trend data history. The data of exact same time slots in former consecutive
            ``trend_len`` weeks (every seven days) will be used as trend history.
        target_length(int):The numbers of steps that need prediction by one piece of history data. Have to be 1 now.
            Default: 1 default:1.
        daily_slots(int): The number of records of one day. Calculated by 24 * 60 /time_fitness. default:24.
    '''
    def __init__(self, closeness_len, period_len, trend_len, target_length=1, daily_slots=24):
        self._c_t = closeness_len
        self._p_t = period_len
        self._t_t = trend_len
        self._target_length = target_length
        self._daily_slots = daily_slots

        # 1 init Move_Sample object
        self.move_sample_closeness = MoveSample(feature_step=self._c_t, feature_stride=1,
                                                feature_length=1, target_length=self._target_length)

        self.move_sample_period = MoveSample(feature_step=self._p_t + 1, feature_stride=int(self._daily_slots),
                                             feature_length=1, target_length=0)

        self.move_sample_trend = MoveSample(feature_step=self._t_t + 1, feature_stride=int(self._daily_slots) * 7,
                                            feature_length=1, target_length=0)

    def move_sample(self, data):
        '''
        Input data to generate closeness, period, trend features and target vector y.

        Args:
            data(ndarray):Orginal temporal data.
        :return:closeness, period, trend and y matrices.
        :type: numpy.ndarray.
        '''
        # 2 general move sample
        closeness, y = self.move_sample_closeness.general_move_sample(data)
        period, _ = self.move_sample_period.general_move_sample(data)
        trend, _ = self.move_sample_trend.general_move_sample(data)

        # 3 remove the front part
        min_length = min(len(closeness), len(period), len(trend))
        closeness = closeness[-min_length:]
        y = y[-min_length:]
        period = period[-min_length:]
        trend = trend[-min_length:]

        # 4 remove tail of period and trend
        period = period[:, :-1]
        trend = trend[:, :-1]

        if self._c_t and self._c_t > 0:
            things = closeness, [0] + list(range(3, len(closeness.shape))) + [1, 2]
            closeness = np.transpose(closeness, [0] + list(range(3, len(closeness.shape))) + [1, 2])
        else:
            closeness = np.array([])

        if self._p_t and self._p_t > 0:
            period = np.transpose(period, [0] + list(range(3, len(period.shape))) + [1, 2])
        else:
            period = np.zeros(shape=[min_length,closeness.shape[1],0,1])

        if self._t_t and self._t_t > 0:
            trend = np.transpose(trend, [0] + list(range(3, len(trend.shape))) + [1, 2])
        else:
            trend = np.zeros(shape=[min_length,closeness.shape[1],0,1])

        y = np.transpose(y, [0] + list(range(2, len(y.shape))) + [1])

            # 拼接 trend, period 和 closeness
        if trend.size == 0:
            combined_features = np.concatenate([period, closeness], axis=-2)
        elif period.size == 0:
            combined_features = np.concatenate([trend, closeness], axis=-2)
        elif closeness.size == 0:
            combined_features = np.concatenate([trend, period], axis=-2)
        else:
            combined_features = np.concatenate([trend, period, closeness], axis=-2)
        combined_features = np.squeeze(combined_features, axis=-1)
        # y = np.squeeze(y, axis = -1)
        return combined_features, y

        # return closeness, period, trend, y

'''
Name: Tensor-Time-Series Dataset Manager:
Param:
    -
Method:
    -
'''
class TTS_DatasetManager:
    def __init__(self, pkl_path:str, his_len:int, pred_len:int,
                 test_ratio=0.1, valid_ratio=0.1, seed=2024, data_mode:int=0) -> None:
        random.seed(seed)
        self.his_len = his_len
        self.pred_len = pred_len
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        # TTS format:
        # shape = (t, dim1, dim2)
        self.set_data_mode(data_mode)
        # self.data = self.data_pkl['data']
        # data_shape = self.data.shape
        # self.time_range = data_shape[0]
        # self.dim1_range = data_shape[1]
        # self.dim2_range = data_shape[2]
        # self.sample = 1
        # split data
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        data_index = list(range(int(self.time_range)-(his_len+pred_len)))
        # random.shuffle(data_index)
        train_index_end = int(len(data_index)*train_ratio)
        valid_index_end = int(len(data_index)*valid_ratio) + train_index_end
        self.trainset = data_index[:train_index_end]
        random.shuffle(self.trainset)
        self.validset = data_index[train_index_end:valid_index_end]
        random.shuffle(self.validset)
        self.testset  = data_index[valid_index_end:]
        random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }
        # print(len(self.trainset), len(self.validset), len(self.testset));exit()

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_his_pred_from_idx(self, idx:int):
        # TODO: check if idx is valid
        his = self.data[idx: int(idx+self.his_len)]
        pred = self.data[int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        return his, pred
    
    def get_seq_from_idx(self, idx:int):
        win = int(self.his_len + self.pred_len)
        data = self.data[idx: idx+win]
        return data

    def set_data_mode(self, data_mode:int):
        self.data = self.data_pkl['data']
        # set data mode to change the shape
        # - 0: (time, dim1, dim2)
        # - 1: (time, dim2, dim1)
        # - 2: (time, dim1*dim2, 1) 
        if data_mode == 0:
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]
        elif data_mode == 1:
            self.data = self.data.transpose(0,2,1)
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]
        elif data_mode == 2:
            self.data = self.data.reshape(self.data.shape[0], -1, 1)
            data_shape = self.data.shape
            self.time_range = data_shape[0]
            self.dim1_range = data_shape[1]
            self.dim2_range = data_shape[2]
        # print(f"Data shape: {data_shape}")
        # exit()

    def get_tensor_shape(self):
        return (self.dim1_range, self.dim2_range)
    
    def get_data_shape(self):
        return (self.time_range, self.dim1_range, self.dim2_range)

    def get_normalizer(self, norm='none'):
        if norm == 'none':
            return DoNothing()
        elif norm == 'sklearn':
            train_range = int(self.time_range * self.train_ratio)
            train_data = self.data[:train_range]
            # normalization and inverse
            scaler = Sklearn_StandNormalizer(train_data)
            return scaler
        elif norm == 'std':
            train_range = int(self.time_range * self.train_ratio)
            train_data = self.data[:train_range]
            # normalization and inverse
            scaler = StandNormalizer(train_data)
            return scaler
        else:
            raise ValueError(f'unknown normalizer: {norm}...')

'''
Name: Multivar Dataset Manager:
Dataset format: 
(time_series_num, time_range, dim_range)
    - time_series_num: number of time series
    - time_range: number of time steps
    - dim_range: number of dimensions
'''
class MTS_DatasetManager:
    def __init__(self, pkl_path:str, his_len:int, pred_len:int,
                 test_ratio=0.1, valid_ratio=0.1, seed:int=2024, data_mode:int=0) -> None:
        random.seed(seed)
        self.his_len = his_len
        self.pred_len = pred_len
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        # set data mode
        self.set_data_mode(data_mode)
        # split dataset
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        # default step = 1
        # split:
        # | <--- trian ---> | <- valid ->  | <- test -> |
        data_index = list(range(int(self.time_range)-(his_len+pred_len)))
        train_index_end = int(len(data_index)*train_ratio)
        valid_index_end = int(len(data_index)*valid_ratio) + train_index_end
        self.trainset = data_index[:train_index_end]
        self.validset = data_index[train_index_end:valid_index_end]
        self.testset  = data_index[valid_index_end:]
        # print(len(self.trainset), len(self.validset), len(self.testset));exit()
        # random shuffle the dataset
        random.shuffle(self.trainset)
        random.shuffle(self.validset)
        random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }

    def set_data_mode(self, data_mode:int):
        self.data = self.data_pkl['data']
        # raw data shape: (time, dim1, dim2)
        # set data mode to change the shape
        # - 0: (time, dim1*dim2, 1)
        # - 1: (time, dim1, 1) * dim2
        # - 2: (time, dim2, 1) * dim1
        if data_mode == 0:
            self.data = self.data.reshape(1, self.data.shape[0], -1)
            self.data = np.expand_dims(self.data, axis=-1)
            data_shape = self.data.shape
            self.time_series_num = int(data_shape[0])
            self.time_range = int(data_shape[1])
            self.dim_range = int(data_shape[2])
        elif data_mode == 1:
            self.data = self.data.transpose(2,0,1)
            self.data = np.expand_dims(self.data, axis=-1)
            data_shape = self.data.shape
            self.time_series_num = int(data_shape[0])
            self.time_range = int(data_shape[1])
            self.dim_range = int(data_shape[2])
        elif data_mode == 2:
            self.data = self.data.transpose(1,0,2)
            self.data = np.expand_dims(self.data, axis=-1)
            data_shape = self.data.shape
            self.time_series_num = int(data_shape[0])
            self.time_range = int(data_shape[1])
            self.dim_range = int(data_shape[2])
        # print(f"Data shape: {data_shape}")
        # exit()

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_time_series_num(self):
        return self.time_series_num

    def get_dim_num(self):
        return self.dim_range
    
    def get_data_shape(self):
        return (self.get_time_series_num(), self.time_range, self.dim_range)

    def get_his_pred_from_idx(self, time_series_idx:int, idx:int):
        if time_series_idx >= self.time_series_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.time_series_num}")
        his = self.data[time_series_idx, idx: int(idx+self.his_len)]
        pred = self.data[time_series_idx, int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        return his, pred
    
    def get_seq_from_idx(self, time_series_idx:int ,idx:int):
        if time_series_idx >= self.time_series_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.time_series_num}")
        win = int(self.his_len + self.pred_len)
        data = self.data[time_series_idx, idx: idx+win]
        return data
    
    def get_normalizer(self, norm='none')->list:
        if norm == 'none':
            normalizer_list = []
            for i in range(self.time_series_num):
                normalizer_list.append(DoNothing())
            return normalizer_list
        elif norm == 'sklearn':
            normalizer_list = []
            train_range = int(self.time_range * self.train_ratio)
            train_data  = self.data[:, :train_range]
            for i in range(self.time_series_num):
                train_data_i = train_data[i]
                scaler = Sklearn_StandNormalizer(train_data_i)
                normalizer_list.append(scaler)
            return normalizer_list
        elif norm == 'std':
            normalizer_list = []
            train_range = int(self.time_range * self.train_ratio)
            train_data  = self.data[:, :train_range]
            for i in range(self.time_series_num):
                train_data_i = train_data[i]
                scaler = StandNormalizer(train_data_i)
                normalizer_list.append(scaler)
            return normalizer_list
        else:
            raise ValueError(f'unknown normalizer: {norm}...')
        
# class LagInput_DatasetManager:
        
if __name__ == '__main__':
    import numpy as np
    # 示例数据：一天中每小时的温度记录（24小时）
    data = []
    for i in range(1000):
        data.append(np.ones((2,2))*i)
    data = np.array(data)
    # data = data.T
    # 初始化 ST_MoveSample 类
    closeness_len = 3  # 使用前3个时间槽的数据作为接近性历史
    period_len = 1     # 使用前1天的相同时间槽的数据作为周期性历史
    trend_len = 0      # 使用前1周的相同时间槽的数据作为趋势性历史
    target_length = 5  # 需要预测的时间步数
    daily_slots = 24   # 一天中的记录数

    st_move_sample = ST_MoveSample(closeness_len, period_len, trend_len, target_length, daily_slots)

    # 生成接近性、周期性、趋势性特征和目标向量 y
    X, y = st_move_sample.move_sample(data)

    # print("Closeness Features:\n", closeness[0])
    # print("Period Features:\n", period[0])
    # print("Trend Features:\n", trend[0])
    # print("Input Data:\n", input_data[0])
    # print("Target Vector y:\n", y[0])