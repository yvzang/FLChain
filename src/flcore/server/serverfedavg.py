import torch
from flcore.client.clientFedAvg import *
from copy import deepcopy
import asyncio
from utils.readData import *
from flcore.server.serverbase import ServerBase


class ServerFedAvg(ServerBase):
    def __init__(self, model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate):
        super(ServerFedAvg, self).__init__(model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate)


    def model_aggregation(self):
        '''对所有参与方的参数加和'''
        self.client.server.print_transaction_pool()
        data_hash = [tx.data for tx in self.client.server.trans_pool]
        data = [asyncio.run(self.client.server.get(h)) for h in data_hash]
        print(data)
    
        
