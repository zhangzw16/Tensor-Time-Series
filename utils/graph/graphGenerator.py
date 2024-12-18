import numpy as np
import os
import pickle
import scipy
import scipy.stats
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from datasets.dataset import TTS_DatasetManager


class GraphGenerator:
    def __init__(self, dataset:TTS_DatasetManager) -> None:
        self.dataset = dataset
        self.data = self.dataset.raw_train_data
        self.tensor_shape = self.dataset.get_tensor_shape()
        self.graph_dir = os.path.join(os.path.dirname(self.dataset.pkl_path), 'graph')
        if not os.path.exists(self.graph_dir):
            os.mkdir(self.graph_dir)        

    def gen_graph(self, n_dim:int, graph_init:str, normal=True):
        if graph_init == 'cosine':
            graph = self.cosine_similarity_matrix(n_dim, normal)
        elif graph_init == 'pearson':
            graph = self.pearson_matrix(n_dim, normal)
        elif graph_init == 'random':
            graph = self.random_matrix(n_dim)
        elif graph_init == 'inverse_pearson':
            graph = self.inverse_pearson_matrix(n_dim, normal)
        elif graph_init == 'unit':
            graph = self.unit_matrix(n_dim)
        else:
            raise ValueError(f"graph_init {graph_init} is not supported")
        pickle.dump(graph, open(os.path.join(self.graph_dir, f'{graph_init}_{n_dim}.pkl'), 'wb'))

    def pearson_matrix(self, n_dim:int, normal=True):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        if n_dim == 0:
            self.data = self.data.transpose(0,2,1)
        graph = np.zeros((dim, dim))
        total_num = int(dim*(dim+1)) // 2
        progress_bar = tqdm(total=total_num, desc=f'Generating Pearson Matrix --> {dim}x{dim}')
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
                except KeyboardInterrupt:
                    exit()
                except :
                    p = 0
                graph[i,j] = p
                graph[j,i] = p
                progress_bar.update(1)
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
    
    def unit_matrix(self, n_dim:int):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        graph = np.eye(dim)
        return graph

    def cosine_similarity_matrix(self, n_dim:int, normal=False):
        dim = self.tensor_shape[n_dim]
        if dim <= 1:
            return np.ones((1,1))
        if n_dim == 0:
            self.data = self.data.transpose(0,2,1)
        graph = np.zeros((dim, dim))
        progress_bar = tqdm(total=dim*dim/2+dim, desc=f'Generating Cosine Matrix--> {dim}x{dim}')
        for i in range(dim):
            seq_i = self.data[:, :, i]
            for j in range(i, dim):
                seq_j = self.data[:, :, j]
                if normal:
                    try: 
                        sim = cosine_similarity(seq_i, seq_j)[0][0]
                        sim = (sim+1)/2
                    except KeyboardInterrupt:
                        exit()
                    except:
                        sim = 0.5
                graph[i,j] = sim
                graph[j,i] = sim
                progress_bar.update(1)
        return graph
    
    def load_pkl_graph(self, pkl_path:str):
        graph = pickle.load(open(pkl_path, 'rb'))
        return graph
    
    def is_constant(self, seq):
        return np.all(seq == seq[0])
    
class GraphGeneratorManager:
    def __init__(self, graph_init:str, dataset:TTS_DatasetManager) -> None:
        self.graph_init = graph_init
        self.dataset = dataset
        self.graph_dir = os.path.join(os.path.dirname(self.dataset.pkl_path), 'graph')

    def load_graph(self, n_dim:int, normal=True):
        tensor_shape = self.dataset.get_tensor_shape()
        dim = tensor_shape[n_dim]
        if dim <= 1:
            print(f"dim {dim} is 1, return ones matrix")
            return np.ones((1,1))
        print(f"Loading graph {self.graph_init}_{n_dim}.pkl")
        graph_path = os.path.join(self.graph_dir, f'{self.graph_init}_{n_dim}.pkl')
        graph = pickle.load(open(graph_path, 'rb'))
        return graph


if __name__=='__main__':
    pass