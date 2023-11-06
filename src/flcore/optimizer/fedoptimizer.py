import torch
from torch.optim import Optimizer

class BasedOptimizer(Optimizer):
    def __init__(self, params, lr) -> None:
        defaults = dict(lr=lr)
        super(BasedOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                p.data.add_(other=(p.grad.data), alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])