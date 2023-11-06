from torch.utils.data import DataLoader
from torch.nn.modules import CrossEntropyLoss
from transformer import Transformer
from copy import deepcopy
from threading import Lock
from queue import Queue
import numpy as np
import torch
from flcore.client.clientbase import Client

class ClientFedAvg(Client):
    def __init__(self, id, model, lr, lr_decay, decay_period, device):
        super().__init__(id, model, lr, lr_decay, decay_period, device)
