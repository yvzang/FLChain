import torch
from flcore.client.clientbase import *
from copy import deepcopy
from torch.nn import Softmax
from utils.readData import *
from flcore.server.serverbase import ServerBase


class ServerALW(ServerBase):
    def __init__(self, model, device, testloader, proxyloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate):
        super(ServerALW, self).__init__(model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate)
        self.proxyloader = proxyloader
        self.fi1 = 0.8
        self.fi2 = 3

    def get_updated_module_with_gama(self, arg_module : torch.nn.Module, gradient : dict, gama, retain_graph=True) -> torch.nn.Module:
        r"""Calculates updated module for provided module by providing the gradient.
        The module parameters can contain grad if provide retrain_graph true.
        
        Args:
            arg_module (dict): parameters of target module need to be updated
            gradient (dict): a gradient need be updated.

        Returns:
            updated module parameters.
        """
        updated_model_dict = {}
        for name, param in arg_module.named_parameters():
            if retain_graph:
                updated_model_dict[name] = gama * (param.data - gradient[name])
            else:
                updated_model_dict[name] = (gama * (param.data - gradient[name])).detach()
        arg_module.set_parameter_state_dict(updated_model_dict)
        return arg_module


    def aggregate_parameters_softmax(self, pseudo_params: list, weights: list, prefix: list = None) -> dict:
        if len(pseudo_params) == 0:
            raise Exception("The aggregated parameters is empty.")
        agg_params = {}
        softmax = Softmax(dim=0)
        cohot_len = len(pseudo_params)
        for key, value in pseudo_params[0].items():
            agg_params[key] = torch.zeros_like(value, device=self.device, requires_grad=True).float()
            for i in range(cohot_len):
                agg_params[key] = agg_params[key] + (softmax(weights)[i] * pseudo_params[i][key].data)
        return agg_params

    
    def __calculate_optim_weight__(self, aggregate_list, weight_list, lr=0.005, epoch=20):
        agg_weight = torch.tensor([torch.log(torch.tensor(w)) for w in weight_list], device=self.device, requires_grad=True)
        gama = torch.tensor(1.0, device=self.device, requires_grad=True)
        softmax = Softmax(dim=0)
        optimizees_list = [ agg_weight]
        opt = torch.optim.SGD(optimizees_list, lr=lr)
        updated_model = deepcopy(self.model)
        updated_model.train()
        for i in range(epoch):
            for image, label in self.proxyloader:
                agg_grad = self.aggregate_parameters_softmax(aggregate_list, agg_weight)
                updated_model = self.get_updated_module_with_gama(updated_model, agg_grad, gama)
                image = self.__to_device__(image); label = self.__to_device__(label)
                opt.zero_grad()
                output = updated_model(image)
                loss = self.loss_fn(output, label)
                loss.backward()
                opt.step()
            print(softmax(agg_weight))
            print(gama)
        return [w for w in softmax(agg_weight).detach()], gama



    def model_aggregation(self, params_queue):
        #对所有参与方的参数加和
        if params_queue.empty():
            raise Exception("没有能进行加和的参数..")
        #先取出所有梯度
        params_list = []
        while(params_queue.empty() == False):
            item = params_queue.get()
            param = item["params"]
            params_list.append(param)
        aggregate_list = params_list
        weight_list = [1 / len(aggregate_list)] * len(aggregate_list)
        weight_list, gama = self.__calculate_optim_weight__(aggregate_list, weight_list)
        print(weight_list)
        #聚合
        agg_gradient = self.aggregate_parameters(aggregate_list, weight_list)
        self.model = self.get_updated_module_with_gama(self.model, agg_gradient, gama, retain_graph=False)
        return self.model.state_dict()
        
