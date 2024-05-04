from .model_base import ModelBase
import importlib
import os

class ModelManager:
    def __init__(self) -> None:
        self.model_name = __name__
    def get_model_class(self, name:str)->ModelBase:
        try:
            model_pkg_path = f'{self.model_name}.{name}.model'
            model_pkg = importlib.import_module(model_pkg_path)
            model_class_name = f'{name}_TensorModel'
            model_class = getattr(model_pkg, model_class_name)
            return model_class
        except:
            raise ImportError(f'can not get model_class: {model_pkg_path}, {name}')
    
