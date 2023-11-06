import torch
from flcore.client.clientbase import *
from copy import deepcopy
import queue
import math
from utils.readData import *
from flcore.server.serverbase import ServerBase


class ServerALW(ServerBase):
    def __init__(self, model, device, testloader, proxyloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate):
        super(ServerALW, self).__init__(model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate)
        self.proxyloader = proxyloader
        self.fi1 = 0.8
        self.fi2 = 3

    def get_grad_loss(self, gradient : dict):
        updated_model = self.get_updated_module(deepcopy(self.model), gradient)
        with torch.no_grad():
            loss = self.__to_device__(torch.zeros(1))
            updated_model.eval()
            for image, label in self.proxyloader:
                image = self.__to_device__(image)
                label = self.__to_device__(label)
                output = updated_model(image)
                #计算损失
                curr_loss = self.loss_fn(output, label)
                loss += curr_loss
            return loss

    
    def __calculate_optim_weight__(self, aggregate_list, weight_list, lr=0.008, bios=0.001):
        if len(aggregate_list) != len(weight_list):
            raise Exception("参数列表和权重列表不一致.")
        for i in range(len(aggregate_list)):
            weight = weight_list[i]
            print("init weight: ", weight)
            #获得初始化的损失
            aggregated = self.aggregate_parameters(aggregate_list, weight_list)
            init_loss = self.get_grad_loss(aggregated)
            print("init loss: ", init_loss)
            init_loss = self.get_grad_loss(aggregated)
            print("init loss: ", init_loss)
            #获得聚合后的损失
            t_weight_list = weight_list[:]; t_weight_list[i] = weight + bios
            aggregated = self.aggregate_parameters(aggregate_list, t_weight_list)
            agg_loss = self.get_grad_loss(aggregated)
            print("after loss: ", agg_loss)
            #更新损失
            update = -lr * (agg_loss.item() - init_loss.item()) / bios
            weight = weight + update
            if weight < 0: weight = 0
            weight_list[i] =weight
            print("after weight: ", weight)
        return aggregate_list, weight_list



    def model_aggregation(self, params_queue):
        #对所有参与方的参数加和
        if params_queue.empty():
            raise Exception("没有能进行加和的参数..")
        with torch.set_grad_enabled(False):
            #先取出所有梯度
            params_list = []
            loss_decent_list = []
            while(params_queue.empty() == False):
                item = params_queue.get()
                param = item["params"]
                params_list.append(param)
            print(self.get_loss())
            print(self.get_loss())
            print(loss_decent_list)
            aggregate_list = params_list
            weight_list = [1 / len(aggregate_list)] * len(aggregate_list)
            decay = 0.98
            for i in range(0, 20):
                aggregate_list, weight_list = self.__calculate_optim_weight__(aggregate_list, weight_list, lr=0.0002*(math.pow(decay, i)))
                print(weight_list)
        if sum(weight_list) < self.fi1: weight_list = [self.fi1 * w / sum(weight_list) for w in weight_list]
        elif sum(weight_list) > self.fi2 : weight_list = [self.fi2 * w / sum(weight_list) for w in weight_list]
        #聚合
        self.model = self.update_parameter_layer(self.model, params_list, weight_list)
        return self.model.state_dict()
        
