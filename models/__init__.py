from .model_base import ModelBase
import importlib
import os
import yaml

class ModelManager:
    def __init__(self) -> None:
        self.model_name = __name__
        self.model_type_map = {
            'Tensor': 'TensorModel',
            'MultiVar': 'MultiVarModel',
            'Stat': 'StatModel',
        }
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.model_registry = yaml.safe_load(open(os.path.join(current_path, 'ModelRegistry.yaml'), 'r'))
    
    def get_model_type(self, model_name:str)->str:
        model_map = self.model_registry['ModelType']
        model_type = None
        for type_name in model_map:
            if model_name in model_map[type_name]:
                model_type = type_name
                break
        return model_type
    
    def is_prior_graph(self, model_name:str)->bool:
        graph_map = self.model_registry['GraphModel']
        if model_name in graph_map['prior']:
            return True
        else:
            return False

    def get_model_class(self, name:str)->ModelBase:
        try:
            model_pkg_path = f'{self.model_name}.{name}.model'
            model_pkg = importlib.import_module(model_pkg_path)
            model_type = self.get_model_type(name)
            model_class_type = self.model_type_map[model_type]
            model_class_name = f'{name}_{model_class_type}'
            model_class = getattr(model_pkg, model_class_name)
            return model_class
        except ImportError as err_info:
            raise ImportError(f'can not get model_class: {model_pkg_path}, {name}, error: {err_info}')
    
