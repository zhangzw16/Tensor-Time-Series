import numpy as np
import pickle
import scipy
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity

from datasets.dataset_tts import TTS_Dataset


class GraphGenerator:
    def __init__(self, dataset:TTS_Dataset, ratio:float=0.3, max_sample:int=3000) -> None:
        self.dataset = dataset
        self.tensor_shape = self.dataset.get_tensor_shape()
        self.data = self.dataset.data
        time_range = self.data.shape[0]
        sample_n = min(int(ratio*time_range), max_sample)
        indices = np.random.choice(time_range, sample_n, replace=False)
        self.data = self.data[indices]

    def pearson_matrix(self, n_dim:int, normal=True):
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
                try:
                    if self.is_constant(seq_i) or self.is_constant(seq_j):
                        p = 0
                    else:
                        p = scipy.stats.pearsonr(seq_i, seq_j)[0]
                    if normal:
                        p = np.abs(p)
                except:
                    p = 0
                graph[i,j] = p
                graph[j,i] = p
        return graph

    def inverse_pearson_matrix(self, n_dim:int, normal=True):
        graph = self.pearson_matrix(n_dim, normal)
        graph = 1 - graph
        return graph

    def random_matrix(self, n_dim:int):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        graph = np.random.rand(dim, dim)
        graph_upper = np.triu(graph, 1)
        graph = graph_upper + graph_upper.T
        for i in range(dim):
            graph[i,i] = 1.0
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
                if normal:
                    try: 
                        sim = cosine_similarity(seq_i, seq_j)[0][0]
                        sim = (sim+1)/2
                    except:
                        sim = 0.5
                graph[i,j] = sim
                graph[j,i] = sim
        return graph
    
    def load_pkl_graph(self, pkl_path:str):
        graph = pickle.load(open(pkl_path, 'rb'))
        return graph
    
    def is_constant(self, seq):
        return np.all(seq == seq[0])
    
class GraphGeneratorManager:
    def __init__(self, graph_init:str, dataset:TTS_Dataset, ratio:float=0.3, max_sample:int=3000) -> None:
        self.graph_generator = GraphGenerator(dataset, ratio, max_sample)
        self.graph_init = graph_init
    def gen_graph(self, n_dim:int, normal=False):
        if self.graph_init == 'cosine':
            return self.graph_generator.cosine_similarity_matrix(n_dim, normal)
        elif self.graph_init == 'pearson':
            return self.graph_generator.pearson_matrix(n_dim, normal)
        elif self.graph_init == 'random':
            return self.graph_generator.random_matrix(n_dim)
        elif self.graph_init == 'inverse_pearson':
            return self.graph_generator.inverse_pearson_matrix(n_dim, normal)
        else:
            return self.graph_generator.random_matrix(n_dim)
    