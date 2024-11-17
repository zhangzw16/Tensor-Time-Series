import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from models.model_base import ModelBase
from models import ModelManager

class LRFinder_Manager:
    def __init__(self, model_name:str, model_configs:dict, trainloader, valloader, output_dir:str, normalizer, device:str='cuda') -> None:
        # basic configs
        self.model_name = model_name
        self.model_configs = model_configs
        self.trainloader = trainloader
        self.valloader = valloader
        self.output_dir = output_dir
        self.normalizer = normalizer
        self.device = device
        # init model
        model_manager = ModelManager()
        self.model = model_manager.get_model_class(self.model_name)(model_configs)
        self.history = {"lr": [], "mean_lr": [], "loss": []}
        self.best_loss = None

    def _get_current_lr(self):
        return [param_group['lr'] for param_group in self.model.optim.param_groups]
    
    def _set_lr(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self._get_current_lr())
        if len(new_lrs) != len(self._get_current_lr()):
            raise ValueError("Length of new_lr must be the same as the current learning rate")
        for param_group, lr in zip(self.model.optim.param_groups, new_lrs):
            param_group['lr'] = lr

    def search_lr(self, lr_start=1e-4, lr_end=1e-2, num_iter=64, smooth_f=0.05, diverge_th=5, diverge_paience_thres=3):
        self.lr_start = lr_start
        self.lr_end = lr_end
        # calculate the multiple of lr
        self.mult = (lr_end/lr_start)**(1/num_iter)
        # init optimizer with lr_start
        self._set_lr(lr_start)

        # initialization
        n_iter = 0
        smoothed_loss = 0.0
        diverge_paience = 0

        # model to device
        self.model.set_device(self.device)
        with tqdm(total=num_iter, desc="Searching optimal learning rate") as pbar:
            while n_iter < num_iter:
                for seq in self.trainloader:
                    n_iter += 1
                    current_lr = self._get_current_lr()
                    # trainning
                    self.model.train()
                    # normalize the input
                    norm_seq = self.normalizer.transform(seq)
                    norm_seq = norm_seq.to(self.device)
                    # forward and get loss
                    norm_pred, norm_truth = self.model.forward(norm_seq)
                    loss = self.model.get_loss(norm_pred, norm_truth)
                    # backward
                    self.model.backward(loss)
                        
                    # smooth the loss
                    if smoothed_loss == 0.0:
                        smoothed_loss = loss.item()
                    else:
                        smoothed_loss = smooth_f * loss.item() + (1 - smooth_f) * smoothed_loss

                    # record the loss
                    mean_lr=sum(current_lr)/len(current_lr)
                    self.history["lr"].append(current_lr)
                    self.history["mean_lr"].append(mean_lr)
                    self.history["loss"].append(smoothed_loss)

                    # check diverge
                    if self.best_loss is None:
                        self.best_loss = smoothed_loss
                    else:
                        if smoothed_loss > diverge_th * self.best_loss:
                            diverge_paience += 1
                        if smoothed_loss < self.best_loss:
                            self.best_loss = smoothed_loss
                            diverge_paience = 0
                    if diverge_paience > diverge_paience_thres:
                        return
                    
                    # update learning rate
                    new_lr = [lr * self.mult for lr in current_lr]
                    self._set_lr(new_lr)

                    # 更新进度条后缀信息
                    pbar.set_postfix(n_iter=n_iter, loss=smoothed_loss, lr=mean_lr)
                    pbar.update(1)
                    if n_iter >= num_iter:
                        break
                if n_iter >= num_iter:
                        break
                
    def save_plot(self):
        plt.figure()
        loss = np.log10(self.history["loss"])
        plt.plot(self.history["mean_lr"], loss)
        plt.xlim(self.lr_start, self.lr_end+0.01)

        min_loss = min(self.history["loss"])
        min_loss_idx = self.history["loss"].index(min_loss)
        best_lr = self.history["mean_lr"][min_loss_idx]
        plt.scatter([best_lr], [np.log10(min_loss)], color='red')
        plt.text(best_lr, np.log10(min_loss), f'LR: {best_lr:.2e}\nLoss: {min_loss:.2e}', color='red')

        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss(log)")
        plt.savefig(os.path.join(self.output_dir, "lr_finder.png"))
        plt.close()

    def get_best_mean_lr(self):
        best_mean_lr = self.history["mean_lr"][self.history["loss"].index(min(self.history["loss"]))]
        return best_mean_lr
    
    def get_best_lr(self):
        best_lr = self.history["lr"][self.history["loss"].index(min(self.history["loss"]))]
        return best_lr
    
    def set_optim_with_lr(self, model:ModelBase, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(model.optim.param_groups)
        if len(new_lrs) != len(self._get_current_lr()):
            raise ValueError("Length of new_lr must be the same as the current learning rate")
        for param_group, lr in zip(model.optim.param_groups, new_lrs):
            param_group['lr'] = lr

        