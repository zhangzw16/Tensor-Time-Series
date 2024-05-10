import numpy as np
import pickle
import scipy
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity

from datasets.dataset import TTS_Dataset


class GraphGenerator:
    def __init__(self, dataset:TTS_Dataset, ratio:float=0.3, max_sample:int=3000) -> None:
        self.dataset = dataset
        self.tensor_shape = self.dataset.get_tensor_shape()
        self.data = self.dataset.data
        time_range = self.data.shape[0]
        sample_n = min(int(ratio*time_range), max_sample)
        indices = np.random.choice(time_range, sample_n, replace=False)
        self.data = self.data[indices]

    def pearson_matrix(self, n_dim:int, normal=False):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        if n_dim == 0:
            self.data = self.data.transpose(0,2,1)
        graph = np.zeros((dim, dim))
        for i in range(dim):
            seq_i = self.data[:, :, i].flatten()
            for j in range(i, dim):
                seq_j = self.data[:, :, j].flatten()
                p = scipy.stats.pearsonr(seq_i, seq_j)[0]
                if normal:
                    try: 
                        p = p+1/2
                    except:
                        p = 0.5
                graph[i,j] = p
                graph[j,i] = p
        return graph

    def cosine_similarity_matrix(self, n_dim:int, normal=False):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        if n_dim == 0:
            self.data = self.data.transpose(0,2,1)
        graph = np.zeros((dim, dim))
        for i in range(dim):
            seq_i = self.data[:, :, i]
            for j in range(i, dim):
                seq_j = self.data[:, :, j]
                sim = cosine_similarity(seq_i, seq_j)[0][0]
                if normal:
                    try: 
                        sim = sim+1/2
                    except:
                        sim = 0.5
                graph[i,j] = sim
                graph[j,i] = sim
        return graph
    
    def load_pkl_graph(self, pkl_path:str):
        graph = pickle.load(open(pkl_path, 'rb'))
        return graph