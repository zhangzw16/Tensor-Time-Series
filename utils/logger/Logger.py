import wandb
import yaml
import os

class LoggerBase:
    def __init__(self, dir:str, project:str, name:str) -> None:
        pass

    def init(self):
        pass

    def log(self, info:dict):
        pass

    def close(self):
        pass

class Logger_none(LoggerBase):
    def __init__(self, dir: str, project: str, name: str, configs:dict) -> None:
        super().__init__(dir, project, name)
        self.log_dir = dir
        self.project = project
        self.name = name
        self.configs = configs
    def init(self):
        print(f'logger-none init:')
        print(f"project: {self.project}")
        print(f"name: {self.name}")

    def log(self, info: dict):
        for i in info:
            print(f'{i}: {info[i]:.3f}')

    def close(self):
        print('Logger_none: Goodbye...')

class Logger_wandb(LoggerBase):
    def __init__(self, dir:str, project:str, name:str, configs:dict) -> None:
        super().__init__(dir, project, name)
        self.log_dir = os.path.join(dir, 'wandb_log')
        self.project_name = project
        self.run_name = name
        self.configs = configs
    def init(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger =wandb.init(
            project=self.project_name, name=self.run_name,
            config=self.configs, dir=self.log_dir)
    
    def log(self, info: dict):
        if self.logger is not None:
            self.logger.log(info)
    
    def close(self):
        if self.logger is not None:
            self.logger.finish()

class LoggerManager:
    def __init__(self) -> None:
        self.log_map = {
            'none': Logger_none,
            'wandb': Logger_wandb,
        }

    def init_logger(self, logger_name:str, dir:str, project:str, name:str, configs:dict):
        logger = self.log_map[logger_name]
        logger = logger(dir=dir, project=project, name=name, configs=configs)
        return logger
    
