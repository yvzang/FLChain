import torch
from flcore.client.clientFedAvg import *
from copy import deepcopy
import asyncio
from utils.readData import *
from flcore.server.serverbase import ServerBase


class ServerFedAvg(ServerBase):
    def __init__(self, model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate):
        super(ServerFedAvg, self).__init__(model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate)

    
        
