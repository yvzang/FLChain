import torch
from flcore.client import *
from torch.nn.modules import CrossEntropyLoss
import random
from threading import Thread, Lock
from copy import deepcopy
import queue
import pandas as pd
from utils.readData import *


class ServerBase():
    def __init__(self, model, device, testloader, testbatchsize, clients, test_epoch, train_round, save_path, agg_rate):
        self.device : str = device
        self.model : torch.nn.Module = self.__to_device__(model)
        self.clients = clients
        self.testloader = testloader
        self.testbatchsize = testbatchsize
        self.loss_fn :  CrossEntropyLoss = self.__to_device__(CrossEntropyLoss())
        self.test_epoch : int = test_epoch
        self.train_round : int = train_round
        self.save_path : str = save_path
        self.agg_rate : float = agg_rate
        self.learning_rate : float = 1
        self.optim : torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def __to_device__(self, module):
        if self.device == 'cpu':
            return module.cpu()
        elif self.device == 'cuda':
            return module.cuda()
        

    def print_percent(self, percent):
        taltol_length = 100
        shap_num = int(percent * taltol_length)
        line_num = taltol_length - shap_num
        _format_shap = "#" * shap_num
        _format_shap = _format_shap + "%" + str(percent.item() * 100)
        _formate_line = "-" * line_num
        print(_format_shap + _formate_line)
        

    def train(self):
        '''训练模型参数'''
        if(len(self.clients) == 0):
            raise RuntimeError("客户端未连接")
        #进行epoch轮迭代
        self.total_epoch = 1
        while(True):
            #挑选train_rate * len(clients)个参与方
            choiced_clients = random.sample(self.clients, int(self.agg_rate * len(self.clients)))
            
            params_queue = queue.Queue(maxsize=len(self.clients))
            lock = Lock()
            thread_list = []
            print("第{}轮训练：".format(self.total_epoch))
            for single_client in choiced_clients:
                #同步训练，获得局部模型参数
                thread = Thread(target=single_client.update_parameters, args=(params_queue, lock), daemon=True, name="{}".format(single_client.client_id))
                thread_list.append(thread)
                thread.start()
            #等待参与方训练完成
            for t in thread_list:
                t.join()
            #加权平均
            global_params = self.model_aggregation(params_queue)
            loss, accuracy = self.get_loss_accu()
            print("loss: {}".format(loss.item()) + ", accuracy: {}".format(accuracy.item()))
            #记录数据
            data = pd.DataFrame([[self.total_epoch, accuracy.item(), loss.item()]], columns=["step", "accuracy", "loss"])
            if self.total_epoch == 1:
                data.to_csv(self.save_path, index=False)
            else:
                data.to_csv(self.save_path, mode="a", index=False, header=False)
            self.set_clients_parameters(self.clients, deepcopy(global_params))
            #判断是否结束训练
            self.total_epoch += 1
            if(self.total_epoch % self.test_epoch == 0):self.print_percent(accuracy)
            if(self.total_epoch == self.train_round):break

    
    def get_loss(self, params=None, fast=False):
        loss, accu = self.get_loss_accu(params, fast)
        return loss
    
    def get_loss_accu(self, params=None, fast=False):
        with torch.no_grad():
            module = deepcopy(self.model)
            if params != None:
                module.load_state_dict(params)
            dataloader = self.testloader if fast==False else self.testfastloader
            loss = self.__to_device__(torch.zeros(1))
            accuracy = self.__to_device__(torch.zeros(1))
            module.eval()
            for image, label in dataloader:
                image = self.__to_device__(image)
                label = self.__to_device__(label)
                output = module(image)
                #计算损失
                curr_loss = self.loss_fn(output, label)
                loss += curr_loss
                #计算准确率
                accu_list = output.argmax(1)
                accu_list = (accu_list == label).sum()
                accuracy += accu_list.float()
            mean_accu = accuracy / (len(dataloader) * self.testbatchsize)
            return loss, mean_accu
        
    def __param_operat__(self, param1 : dict, param2 : dict, op : str):
        with torch.no_grad():
            result = {}
            for param_name in param1.keys():
                if op in '+':
                    result[param_name] = param1[param_name] + param2[param_name]
                elif op in '-':
                    result[param_name] = param1[param_name] - param2[param_name]
                elif op in '*':
                    if isinstance(param1, dict):
                        result[param_name] = param1[param_name] * param2[param_name]
                    else:
                        result[param_name] = param1 * param2[param_name]
            return result
        

    def get_updated_module(self, arg_module : torch.nn.Module, gradient : dict) -> torch.nn.Module:
        r"""Calculates updated module for provided module by providing the gradient.
        The module parameters can contain grad if provide retrain_graph true.
        
        Args:
            arg_module (dict): parameters of target module need to be updated
            gradient (dict): a gradient need be updated.

        Returns:
            updated module parameters.
        """
        for name, param in arg_module.named_parameters():
            param.data -= gradient[name].data
        return arg_module
    

    def aggregate_parameters(self, pseudo_params: list, weights: list, prefix : list = None) -> dict:
        r"""Aggregates parameters with provided weights

        Args:
            pseudo_params (list[dict]): a list containing all parameters updated by clients.
            weights (list[dict]): a weight vector containing all weights aggregateing parameters.

        returns:
            a aggregated result.
        """
        if len(pseudo_params) == 0:
            raise Exception("The aggregated parameters is empty.")
        elif len(pseudo_params) != len(weights):
            raise Exception("The lengh of aggerated gradients weights list is not equal.")
        agg_params = {}
        #聚合
        for param_name, param in pseudo_params[0].items():
            if prefix == None or param_name in prefix:
                agg_params[param_name] = torch.zeros_like(param).float()
                for grad, weight in zip(pseudo_params, weights):
                    if isinstance(weight, dict):
                        agg_params[param_name] = agg_params[param_name] + weight[param_name].data * grad[param_name].data
                    else:
                        agg_params[param_name] = agg_params[param_name] + weight * grad[param_name].data
        return agg_params
    
    
    def update_parameter_layer(self, arg_module : torch.nn.Module, psu_grad : list, weights : list):
        r"""Updates provided module parameters with pseudo gradients and weights list.

        Args:
            arg_module (torch.nn.Module): original module need to be updated.
            psu_grad (list[dict]): weighted aggregated pseudo gradients list.
            weights (list[dict]): weight list.
            retain_graph (bool): a flag to retain calculation graph.

        Returns:
            a result module (torch.nn.Module).
        """
        agg_module = deepcopy(arg_module)
        agg_grad = self.aggregate_parameters(psu_grad, weights)
        agg_module = self.get_updated_module(arg_module, agg_grad)
        return agg_module
    
    
    def set_buffer_layer(self, arg_module : torch.nn.Module, target):
        r"""Sets module buffer layer with provided target module or parameters.
            Note that the function will modify original provided module parameters.

        Args:
            arg_module (torch.nn.Module): the module need be updated.
            target (torch.nn.Module || dict): the module or parameters provids buffer layer.

        Returns:
            updated module.
        """
        if isinstance(target, torch.nn.Module):
            buffer_dict = {name: param for name, param in target.named_buffers()}
        elif isinstance(target, dict):
            buffer_dict = {name: target[name] for name, _ in arg_module.named_buffers()}
        arg_module._set_buffers_(buffer_dict)
        return arg_module
    

    def model_aggregation(self, params_queue : queue.Queue):
        '''对所有参与方的参数加和'''
        if params_queue.empty():
            raise Exception("没有能进行加和的参数..")
        #先取出所有梯度
        item = params_queue.get()
        param = item["params"]
        self.model = self.update_parameter_layer(self.model, [param], [1.0])
        return self.model.state_dict()
    
        

    def set_clients_parameters(self, clients, para):
        '''将参数para传递给参与方列表clients'''
        for client in clients:
            client.set_parameters(para)
