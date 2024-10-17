import os
import yaml
import pickle
import numpy as np
from models import ModelManager
from datasets.dataset import TTS_Dataset
from datasets.dataloader import TTS_DataLoader
from utils.graph.graphGenerator import GraphGeneratorManager

time_list = [512, 1024, 2048]
dim1_list = [1, 16, 64, 512]
dim2_list = [1, 8, 16, 32]
his_len = [12, 48, 96]
pred_len = [3, 6, 12, 24]

class ModelTester:
    def __init__(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        self.ensure_test_data()
        self.model_manager = ModelManager()

    def ensure_test_data(self):
        if not os.path.exists(self.test_data_path):
            os.makedirs(self.test_data_path)
        for time in time_list:
            for dim1 in dim1_list:
                for dim2 in dim2_list:
                    if not os.path.exists(os.path.join(self.test_data_path, f'{time}_{dim1}_{dim2}.pkl')):
                        self.generate_data(time, dim1, dim2)

    def generate_data(self, time, dim1, dim2):
        data = np.random.rand(time, dim1, dim2)
        pkl_data = {'data': data}
        pkl_path = os.path.join(self.test_data_path, f'{time}_{dim1}_{dim2}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def get_base_dir():
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def test_model(self, model_name:str):
        model_type = self.model_manager.get_model_type(model_name)
        if model_type != 'Tensor':
            raise ValueError(f"Model type {model_type} is not supported.")
        print(f"Model: {model_name}")
        for time in time_list:
            for dim1 in dim1_list:
                for dim2 in dim2_list:
                    pkl_path = os.path.join(self.test_data_path, f'{time}_{dim1}_{dim2}.pkl')
                    for h_len in his_len:
                        for p_len in pred_len:
                            result = self.tensor_model_test(model_name, pkl_path, h_len, p_len)
                            print(f"{his_len}-{pred_len}-({time}, {dim1}, {dim2}): {result}")
    
    def tensor_model_test(self, model_name:str, pkl_path:str, his_len:int, pred_len:int):
        base_dir = self.get_base_dir()
        config = yaml.safe_load(open(os.path.join(base_dir, 'tasks', 'tensor_tasks_template.yaml'), 'r'))
        device = 'cuda'
        batch_size = 8

        # dataset
        dataset = TTS_Dataset(pkl_path, his_len, pred_len, 0.1, 0.1, 0)
        dataloader = TTS_DataLoader(dataset, 'train', batch_size=batch_size, drop_last=False)

        # model
        config['device'] = device
        config['his_len'] = his_len
        config['pred_len'] = pred_len
        config['normalizer'] = dataset.get_normalizer(norm='std')
        config['graph_init'] = 'random'
        config['graphGenerator'] = GraphGeneratorManager(config['graph_init'], dataset)
        config['tensor_shape'] = dataset.get_tensor_shape()
        model = self.model_manager.get_model_class(model_name)(config)
        model.set_device(device)

        # test
        model.train()
        for seq, axu_info in dataloader.get_batch(separate=False):
            seq = seq.to(device)
            pred, truth = model(seq, axu_info)
            if pred.shape != truth.shape:
                result = f"Can not match the shape: {pred.shape} != {truth.shape}"
            else:
                result = 'OK'
            # compute the loss & backward
            loss = model.compute_loss(pred, truth)
            loss.backward()
            break
        return result
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='NET3',
                        help="only tensor model")
    args = parser.parse_args()

    model_tester = ModelTester()
    model_tester.test_model(args.name)
