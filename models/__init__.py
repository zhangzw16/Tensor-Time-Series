from .model_base import ModelBase
import importlib
import os

class ModelManager:
    def __init__(self) -> None:
        self.model_name = __name__
        self.model_type_map = {
            'Tensor': 'TensorModel',
            'MultiVar': 'MultiVarModel',
            'Stat': 'StatModel',
        }
    def get_model_class(self, model_type:str, name:str)->ModelBase:
        try:
            model_pkg_path = f'{self.model_name}.{name}.model'
            model_pkg = importlib.import_module(model_pkg_path)
            model_class_type = self.model_type_map[model_type]
            model_class_name = f'{name}_{model_class_type}'
            model_class = getattr(model_pkg, model_class_name)
            return model_class
        except:
            raise ImportError(f'can not get model_class: {model_pkg_path}, {name}')
    
