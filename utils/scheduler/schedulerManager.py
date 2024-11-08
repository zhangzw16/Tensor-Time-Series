import torch.optim.lr_scheduler as lr_scheduler

class FakeScheduler:
    def __init__(self):
        pass
    def step(self):
        return None

class SchedulerManager:
    def __init__(self) -> None:
        self.scheduler_map = {
            'None': self.do_nothing,
            'StepLR': self.StepLR,
            'MultiStepLR': self.MultiStepLR,
            'ExponentialLR': self.ExponentialLR,
            'ReduceLROnPlateau': self.ReduceLROnPlateau,
        }

    def get_scheduler(self, optim, name:str='None'):
        return self.scheduler_map[name](optim)

    def do_nothing(self, optim):
        return FakeScheduler()

    def StepLR(self, optim, step_size:int=20, gamma:float=0.1):
        return lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    
    def MultiStepLR(self, optim, milestones:list=[30,80], gamma:float=0.1):
        return lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=gamma)
    
    def ExponentialLR(self, optim, gamma:float=0.95):
        return lr_scheduler.ExponentialLR(optim, gamma=gamma)
    
    def ReduceLROnPlateau(self, optim, mode:str='min', factor:float=0.9, patience:int=3, threshold:float=1e-5, verbose:bool=False):
        return lr_scheduler.ReduceLROnPlateau(optim, mode=mode, factor=factor, patience=patience, threshold=threshold, verbose=verbose)
    