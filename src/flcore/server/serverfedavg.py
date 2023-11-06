import torch
from flcore.client.clientFedAvg import *
from copy import deepcopy
import queue
from utils.readData import *
from flcore.server.serverbase import ServerBase


class ServerFedAvg(ServerBase):
    def __init__(self, model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate):
        super(ServerFedAvg, self).__init__(model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate)


    def model_aggregation(self, params_queue : queue.Queue):
        '''对所有参与方的参数加和'''
        if params_queue.empty():
            raise Exception("没有能进行加和的参数..")
        with torch.set_grad_enabled(False):
            #先取出所有梯度
            params_list = []
            while(params_queue.empty() == False):
                item = params_queue.get()
                param = item["params"]
                params_list.append(param)
        #聚合
        weights = [1 / len(params_list)] * len(params_list)
        self.model = self.update_parameter_layer(self.model, params_list, weights)
        return self.model.state_dict()
    
        
