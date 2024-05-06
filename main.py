import os
import yaml

from tasks.tensor_task import TensorTask

if __name__ == '__main__':
    # load configs
    yaml_path = ''
    with open(yaml_path, 'r') as file:
        configs = yaml.safe_load(file)
    # load mode
    mode = configs['mode']
    
    # init task
    task = TensorTask(configs)

    # mode = ['train', 'test', 'train-test']
    if mode=='train' or mode == 'train-test':
        task.train()
    
    if mode=='test' or mode == 'train-test':
        res = task.test()

