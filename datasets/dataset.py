import os
import pickle as pkl
import numpy as np
import random

from .normalizer import Sklearn_StandNormalizer, DoNothing, StandNormalizer

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
        combined_features = np.transpose(combined_features, [0, 3, 1, 2])
        y = np.transpose(y, [0, 3, 1, 2])
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

DatasetTemporalResolution = {
    'JONAS_NYC_taxi': 24 * 2,
    'METRO_HZ': 24 * 4
}

class TTS_DatasetManager:
    def __init__(self, pkl_path:str, his_len:int, pred_len:int, normalizer_name:str='none',
                 test_ratio=0.1, valid_ratio=0.1, seed=2024, data_mode:int=0, lag_input:list=[]) -> None:
        random.seed(seed)
        self.pkl_path = pkl_path
        self.his_len = his_len
        self.pred_len = pred_len
        self.normalizer_name = normalizer_name
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        self.dataset_name = pkl_path.split('/')[-2]
        if len(lag_input)==3:
            if self.dataset_name in DatasetTemporalResolution.keys():
                self.time_fitness = DatasetTemporalResolution[self.dataset_name]
            else:
                raise ValueError(f"Unknown dataset temporal resolution: {self.dataset_name}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        self.dataset_name = self.data_pkl.split('/')[-2]
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        # TTS format:
        # shape = (t, dim1, dim2)
        self.make_datasets(data_mode, lag_input)
        # self.data = self.data_pkl['data']
        # data_shape = self.data.shape
        # self.time_range = data_shape[0]
        # self.dim1_range = data_shape[1]
        # self.dim2_range = data_shape[2]
        # self.sample = 1
        # split data
        # data_index = list(range(int(self.time_range)-(his_len+pred_len)))
        # # random.shuffle(data_index)
        # train_index_end = int(len(data_index)*train_ratio)
        # valid_index_end = int(len(data_index)*valid_ratio) + train_index_end
        # self.trainset = data_index[:train_index_end]
        # random.shuffle(self.trainset)
        # self.validset = data_index[train_index_end:valid_index_end]
        # random.shuffle(self.validset)
        # self.testset  = data_index[valid_index_end:]
        # random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }
        # print(len(self.trainset), len(self.validset), len(self.testset));exit()

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_his_pred_from_idx(self, idx:int):
        # his = self.data[idx: int(idx+self.his_len)]
        # pred = self.data[int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        data = self.data[idx]
        his = data[:self.his_len]
        pred = data[self.his_len:]
        return his, pred
    
    def get_seq_from_idx(self, idx:int):
        # win = int(self.his_len + self.pred_len)
        # data = self.data[idx: idx+win]
        data = self.data[idx]
        return data

    def make_datasets(self, data_mode:int, lag_input = []):
        self.data = self.data_pkl['data']
        sample_length = self.his_len + self.pred_len
        sample_num = self.data.shape[0] - sample_length
        train_end = int(sample_num * self.train_ratio)
        valid_end = int(sample_num * self.valid_ratio) + train_end
        self.raw_train_data = self.data[:train_end]
        # normalize data
        self.normalizer = self.init_normalizer(self.normalizer_name, self.raw_train_data)
        self.data = self.normalizer.transform(self.data)
        if len(lag_input)==3:
            self.moving_sampler = ST_MoveSample(closeness_len=lag_input[2], period_len=lag_input[1], trend_len=lag_input[0], target_length=self.pred_len, daily_slots=self.time_fitness)
        # the shape of data read from .pkl is (T, N, M),
        # set data mode to change the shape
        # - 0: Tensor Direct        (T, N, M)
        # - 1: Modality-Independent (T x M, N, 1)
        # - 2: Modality-Individual  (T, N, 1) x M
        if len(lag_input)==3:
            if data_mode == 0:
                self.subset_num = 1
                data_shape = self.data.shape
                self.time_range = data_shape[0]
                self.variable_num = data_shape[1]
                self.modality_num = data_shape[2]
                input, targets = self.moving_sampler.move_sample(self.data)
                sample_num = input.shape[0]
                train_end = int(sample_num * self.train_ratio)
                valid_end = int(sample_num * self.valid_ratio) + train_end
                all_sets = np.concatenate([input, targets], axis=1)
                all_sets = np.expand_dims(all_sets, axis=0)
                self.trainset = all_sets[:, :train_end]
                self.validset = all_sets[:, train_end:valid_end]
                self.testset  = all_sets[:, valid_end:]
            elif data_mode == 1:
                self.subset_num = 1
                data_shape = self.data.shape
                self.time_range = data_shape[0]
                self.variable_num = data_shape[1]
                self.modality_num = 1
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                trainset = []
                validset = []
                testset = []
                for i in range(M):
                    input, targets = self.moving_sampler.move_sample(np.expand_dims(self.data[:,:,i],-1))
                    sample_num = input.shape[0]
                    train_end = int(sample_num * self.train_ratio)
                    valid_end = int(sample_num * self.valid_ratio) + train_end
                    all_sets = np.concatenate([input, targets], axis=1)
                    all_sets = np.expand_dims(all_sets, axis=0)
                    trainset.append(all_sets[:, :train_end])
                    validset.append(all_sets[:, train_end:valid_end])
                    testset.append(all_sets[:, valid_end:])
                self.trainset = np.concatenate(trainset, axis=1)
                self.validset = np.concatenate(validset, axis=1)
                self.testset  = np.concatenate(testset, axis=1)
                # self.trainset = np.array(trainset)
                # self.validset = np.array(validset)
                # self.testset = np.array(testset)
        
        else:
            if data_mode == 0:
                # Tensor Direct
                # (T, N, M) -> (1, T, N, M)
                self.subset_num = 1
                data_shape = self.data.shape
                self.time_range = data_shape[0]
                self.variable_num = data_shape[1]
                self.modality_num = data_shape[2]
                sample_list = []
                for i in range(sample_num):
                    sample_list.append(self.data[i: i+sample_length])
                self.data = np.array(sample_list)
                self.data = np.expand_dims(self.data, axis=0)
                self.trainset = self.data[:, :train_end]
                self.validset = self.data[:, train_end:valid_end]
                self.testset  = self.data[:, valid_end:]
            elif data_mode == 1:
                # Modality-Independent
                # (T, N, M) -> (1, T x M, N, 1)
                # (T, N, M) -> (1,BS， T x M, N, 1)
                # (T, N, M) -> (1, BS*M, T ，N, 1)
                self.subset_num = 1
                data_shape = self.data.shape
                self.time_range = data_shape[0]
                self.variable_num = data_shape[1]
                self.modality_num = 1
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                trainset = []
                validset = []
                testset = []
                # trainset
                for i in range(train_end):
                    for j in range(M):
                        sample = self.data[i:i+sample_length, :, j]
                        trainset.append(sample)
                # validset
                for i in range(train_end, valid_end):
                    for j in range(M):
                        sample = self.data[i:i+sample_length, :, j]
                        validset.append(sample)
                # testset
                for i in range(valid_end, sample_num):
                    for j in range(M):
                        sample = self.data[i:i+sample_length, :, j]
                        testset.append(sample)
                # dataset
                self.trainset = np.array(trainset)
                self.trainset = np.expand_dims(self.trainset, axis=0)
                self.trainset = np.expand_dims(self.trainset, axis=-1)
                self.validset = np.array(validset)
                self.validset = np.expand_dims(self.validset, axis=0)
                self.validset = np.expand_dims(self.validset, axis=-1)
                self.testset  = np.array(testset)
                self.testset  = np.expand_dims(self.testset, axis=0)
                self.testset = np.expand_dims(self.testset, axis=-1)
            elif data_mode == 2:
                # Modality-Individual
                # (T, N, M) -> (M, T, N, 1)
                self.subset_num = int(self.data.shape[2])
                data_shape = self.data.shape
                self.time_range = data_shape[0]
                self.variable_num = data_shape[1]
                self.modality_num = 1
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                trainset = np.zeros((self.subset_num, train_end, sample_length, N))
                validset = np.zeros((self.subset_num, valid_end-train_end, sample_length, N))
                testset  = np.zeros((self.subset_num, sample_num-valid_end, sample_length, N))
                for sub in range(self.subset_num):
                    for i in range(train_end):
                        sample = self.data[i:i+sample_length, :, sub]
                        trainset[sub, i] = sample
                    for i in range(train_end, valid_end):
                        sample = self.data[i:i+sample_length, :, sub]
                        validset[sub, i-train_end] = sample
                    for i in range(valid_end, sample_num):
                        sample = self.data[i:i+sample_length, :, sub]
                        testset[sub, i-valid_end] = sample
                self.trainset = trainset
                self.validset = validset
                self.testset  = testset

    def get_subset_num(self):
        return self.subset_num

    def get_tensor_shape(self):
        return (self.variable_num, self.modality_num)
    
    def get_data_shape(self):
        return (self.time_range, self.variable_num, self.modality_num)

    # def get_normalizer(self, norm='none'):
    #     if norm == 'none':
    #         return DoNothing()
    #     elif norm == 'sklearn':
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data = self.data[:train_range]
    #         # normalization and inverse
    #         scaler = Sklearn_StandNormalizer(train_data)
    #         return scaler
    #     elif norm == 'std':
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data = self.data[:train_range]
    #         # normalization and inverse
    #         scaler = StandNormalizer(train_data)
    #         return scaler
    #     else:
    #         raise ValueError(f'unknown normalizer: {norm}...')
        
    def init_normalizer(self, norm, train_data):
        if norm == 'none':
            return DoNothing()
        elif norm == 'sklearn':
            return Sklearn_StandNormalizer(train_data)
        elif norm == 'std':
            return StandNormalizer(train_data)
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
    def __init__(self, pkl_path:str, his_len:int, pred_len:int, normalizer_name:str='none',
                 test_ratio=0.1, valid_ratio=0.1, seed:int=2024, data_mode:int=0, lag_input:list=[]) -> None:
        random.seed(seed)
        self.his_len = his_len
        self.pred_len = pred_len
        self.normalizer_name = normalizer_name
        # load pkl
        if not os.path.exists(pkl_path):
            raise FileExistsError(f"Can not find file: {pkl_path}")
        self.dataset_name = pkl_path.split('/')[-2]
        if len(lag_input)==3:
            if self.dataset_name in DatasetTemporalResolution.keys():
                self.time_fitness = DatasetTemporalResolution[self.dataset_name]
            else:
                raise ValueError(f"Unknown dataset temporal resolution: {self.dataset_name}")
        with open(pkl_path, 'rb') as file:
            self.data_pkl = pkl.load(file)
        # split dataset
        train_ratio = 1 - test_ratio - valid_ratio
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        if train_ratio < 0:
            raise ValueError(f"invalid ratio. train:{train_ratio}, valid:{valid_ratio}, test:{test_ratio}")
        
        # set data mode
        self.make_datasets(data_mode, normalizer_name, lag_input)

        # random shuffle the dataset
        random.shuffle(self.trainset)
        random.shuffle(self.validset)
        random.shuffle(self.testset)
        self.dataset_map = {
            'train': self.trainset,
            'valid': self.validset,
            'test' : self.testset
        }

    def make_datasets(self, data_mode:int, normalizer_name:str='none', lag_input = []):
        self.data = self.data_pkl['data']
        sample_length = self.his_len + self.pred_len
        sample_num = self.data.shape[0] - sample_length
        train_end = int(sample_num * self.train_ratio)
        valid_end = int(sample_num * self.valid_ratio) + train_end
        self.raw_train_data = self.data[:train_end]
        # normalize data
        self.normalizer = self.init_normalizer(normalizer_name, self.raw_train_data)
        self.data = self.normalizer.transform(self.data)
        if len(lag_input)==3:
            self.moving_sampler = ST_MoveSample(closeness_len=lag_input[2], period_len=lag_input[1], trend_len=lag_input[0], target_length=self.pred_len, daily_slots=self.time_fitness)
        # the shape of data read from .pkl is (T, N, M),
        # set data mode to change the shape
        # - 0: Channel-Modality-Mixing      (T, NxM, 1)       (Train one model)
        # - 1: Channel-Modality-Independent (T x N x M, 1, 1) (Train one model)
        # - 2: Modality-Independent         (T x M, N)        (Trian one model)
        # - 3: Modality-Individual          (T, N) x M        (Train M models)
        if len(lag_input)==3:
            if data_mode == 0:
                self.subset_num = 1
                self.time_range = int(self.data.shape[0])
                self.variable_num = int(self.data.shape[1] * self.data.shape[2])
                self.modality_num = 1
                self.data = self.data.reshape(self.time_range, -1, 1)
                input, targets = self.moving_sampler.move_sample(self.data)
                sample_num = input.shape[0]
                train_end = int(sample_num * self.train_ratio)
                valid_end = int(sample_num * self.valid_ratio) + train_end
                all_sets = np.concatenate([input, targets], axis=1)
                all_sets = np.expand_dims(all_sets, axis=0)
                self.trainset = all_sets[:, :train_end]
                self.validset = all_sets[:, train_end:valid_end]
                self.testset  = all_sets[:, valid_end:]
            elif data_mode == 1:
                self.subset_num = 1
                self.subset_num = 1
                self.time_range = int(self.data.shape[0])
                self.modality_num = 1
                self.variable_num = 1
                self.data = self.data.reshape(self.time_range, 1, -1)
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                trainset = []
                validset = []
                testset = []
                for i in range(M):
                    input, targets = self.moving_sampler.move_sample(np.expand_dims(self.data[:,:,i],-1))
                    sample_num = input.shape[0]
                    train_end = int(sample_num * self.train_ratio)
                    valid_end = int(sample_num * self.valid_ratio) + train_end
                    all_sets = np.concatenate([input, targets], axis=1)
                    all_sets = np.expand_dims(all_sets, axis=0)
                    trainset.append(all_sets[:, :train_end])
                    validset.append(all_sets[:, train_end:valid_end])
                    testset.append(all_sets[:, valid_end:])
                self.trainset = np.concatenate(trainset, axis=1)
                self.validset = np.concatenate(validset, axis=1)
                self.testset  = np.concatenate(testset, axis=1)
            elif data_mode == 2:
                self.subset_num = 1
                self.time_range = int(self.data.shape[0])
                self.modality_num = 1
                self.variable_num = int(self.data.shape[1])
                trainset = []
                validset = []
                testset = []
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                for i in range(M):
                    input, targets = self.moving_sampler.move_sample(np.expand_dims(self.data[:,:,i],-1))
                    sample_num = input.shape[0]
                    train_end = int(sample_num * self.train_ratio)
                    valid_end = int(sample_num * self.valid_ratio) + train_end
                    all_sets = np.concatenate([input, targets], axis=1)
                    all_sets = np.expand_dims(all_sets, axis=0)
                    trainset.append(all_sets[:, :train_end])
                    validset.append(all_sets[:, train_end:valid_end])
                    testset.append(all_sets[:, valid_end:])
                self.trainset = np.concatenate(trainset, axis=1)
                self.validset = np.concatenate(validset, axis=1)
                self.testset  = np.concatenate(testset, axis=1)

            
        else:
            if data_mode == 0:
                # Channel-Modality-Mixing
                # (T, N, M) -> (1, T, NxM, 1)
                self.subset_num = 1
                self.time_range = int(self.data.shape[0])
                self.variable_num = int(self.data.shape[1] * self.data.shape[2])
                self.modality_num = 1
                self.data = self.data.reshape(self.time_range, -1, 1)
                # self.data = np.expand_dims(self.data, axis=-1)
                
                sample_list = []
                for i in range(sample_num):
                    data = self.data[i: i+sample_length]
                    # print(data.shape)
                    sample_list.append(data)
                self.data = np.array(sample_list)
                # self.data = np.concatenate(sample_list, axis=0)
                self.data = np.expand_dims(self.data, axis=0)
                self.trainset = self.data[:, :train_end]
                self.validset = self.data[:, train_end:valid_end]
                self.testset  = self.data[:, valid_end:]
            elif data_mode == 1:
                # Channel-Modality-Independent
                # (T, N, M) -> (1, T x N x M, 1, 1)
                self.subset_num = 1
                self.time_range = int(self.data.shape[0])
                self.modality_num = 1
                self.variable_num = 1
                trainset = []
                validset = []
                testset = []
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                # trainset
                for i in range(train_end):
                    for j in range(N):
                        for k in range(M):
                            sample = self.data[i:i+sample_length, j, k]
                            trainset.append(sample)
                # validset
                for i in range(train_end, valid_end):
                    for j in range(N):
                        for k in range(M):
                            sample = self.data[i:i+sample_length, j, k]
                            validset.append(sample)
                # testset
                for i in range(valid_end, sample_num):
                    for j in range(N):
                        for k in range(M):
                            sample = self.data[i:i+sample_length, j, k]
                            testset.append(sample)
                self.trainset = np.array(trainset)
                self.trainset = np.expand_dims(self.trainset, axis=0)
                self.trainset = np.expand_dims(self.trainset, axis=-1)
                self.validset = np.array(validset)
                self.validset = np.expand_dims(self.validset, axis=0)
                self.validset = np.expand_dims(self.validset, axis=-1)
                # self.trainset = np.expand_dims(self.validset, axis=-1)
                self.testset  = np.array(testset)
                self.testset  = np.expand_dims(self.testset, axis=0)
                self.testset = np.expand_dims(self.testset, axis=-1)
                # self.trainset = np.expand_dims(self.testset, axis=-1)

            elif data_mode == 2:
                # Modality-Independent
                # (T, N, M) -> (1, T x M, N, 1)
                self.subset_num = 1
                self.time_range = int(self.data.shape[0])
                self.modality_num = 1
                self.variable_num = int(self.data.shape[1])
                trainset = []
                validset = []
                testset = []
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                # trainset
                for i in range(train_end):
                    for j in range(M):
                        sample = self.data[i:i+sample_length, :, j]
                        trainset.append(sample)
                # validset
                for i in range(train_end, valid_end):
                    for j in range(M):
                        sample = self.data[i:i+sample_length, :, j]
                        validset.append(sample)
                # testset
                for i in range(valid_end, sample_num):
                    for j in range(M):
                        sample = self.data[i:i+sample_length, :, j]
                        testset.append(sample)
                self.trainset = np.array(trainset)
                self.trainset = np.expand_dims(self.trainset, axis=0)
                self.trainset = np.expand_dims(self.trainset, axis=-1)
                self.validset = np.array(validset)
                self.validset = np.expand_dims(self.validset, axis=0)
                self.validset = np.expand_dims(self.validset, axis=-1)
                self.testset  = np.array(testset)
                self.testset  = np.expand_dims(self.testset, axis=0)
                self.testset = np.expand_dims(self.testset, axis=-1)

            elif data_mode == 3:
                # Modality-Individual
                # (T, N, M) -> (M, T, N, 1)
                self.subset_num = int(self.data.shape[2])
                self.time_range = int(self.data.shape[0])
                self.modality_num = 1
                self.variable_num = int(self.data.shape[1])
                T = self.data.shape[0]
                N = self.data.shape[1]
                M = self.data.shape[2]
                trainset = np.zeros((self.subset_num, train_end, sample_length, N))
                validset = np.zeros((self.subset_num, valid_end-train_end, sample_length, N))
                testset  = np.zeros((self.subset_num, sample_num-valid_end, sample_length, N))
                for sub in range(self.subset_num):
                    for i in range(train_end):
                        sample = self.data[i:i+sample_length, :, sub]
                        trainset[sub, i] = sample
                    for i in range(train_end, valid_end):
                        sample = self.data[i:i+sample_length, :, sub]
                        validset[sub, i-train_end] = sample
                    for i in range(valid_end, sample_num):
                        sample = self.data[i:i+sample_length, :, sub]
                        testset[sub, i-valid_end] = sample
                self.trainset = trainset
                self.validset = validset
                self.testset  = testset

    def get_dataset(self, name:str):
        return self.dataset_map[name]
    
    def get_subset_num(self):
        return self.subset_num

    def get_modality_num(self):
        return self.modality_num
    
    def get_variable_num(self):
        return self.variable_num
    
    def get_tensor_shape(self):
        return (self.variable_num, self.modality_num)

    def get_data_shape(self):
        return (self.time_range, self.variable_num, self.modality_num)

    def get_his_pred_from_idx(self, time_series_idx:int, idx:int):
        if time_series_idx >= self.subset_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.subset_num}")
        # his = self.data[time_series_idx, idx: int(idx+self.his_len)]
        # pred = self.data[time_series_idx, int(idx+self.his_len): int(idx+self.his_len + self.pred_len)]
        data = self.data[time_series_idx, idx]
        his = data[:self.his_len]
        pred = data[self.his_len:]
        return his, pred
    
    def get_seq_from_idx(self, time_series_idx:int ,idx:int):
        if time_series_idx >= self.subset_num:
            raise ValueError(f"Invalid time series index: {time_series_idx}, Max: {self.subset_num}")
        data = self.data[time_series_idx, idx] 
        return data

    def init_normalizer(self, norm, train_data):
        if norm == 'none':
            return DoNothing()
        elif norm == 'sklearn':
            return Sklearn_StandNormalizer(train_data)
        elif norm == 'std':
            return StandNormalizer(train_data)
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

    # def get_normalizer(self, norm='none')->list:
    #     if norm == 'none':
    #         normalizer_list = []
    #         for i in range(self.subset_num):
    #             normalizer_list.append(DoNothing())
    #         return normalizer_list
    #     elif norm == 'sklearn':
    #         normalizer_list = []
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data  = self.data[:, :train_range]
    #         for i in range(self.subset_num):
    #             train_data_i = train_data[i]
    #             scaler = Sklearn_StandNormalizer(train_data_i)
    #             normalizer_list.append(scaler)
    #         return normalizer_list
    #     elif norm == 'std':
    #         normalizer_list = []
    #         train_range = int(self.time_range * self.train_ratio)
    #         train_data  = self.data[:, :train_range]
    #         for i in range(self.subset_num):
    #             train_data_i = train_data[i]
    #             scaler = StandNormalizer(train_data_i)
    #             normalizer_list.append(scaler)
    #         return normalizer_list
    #     else:
    #         raise ValueError(f'unknown normalizer: {norm}...')
