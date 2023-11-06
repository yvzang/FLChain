
from torch.utils.data import DataLoader
from torch.nn.modules import CrossEntropyLoss
from transformer import Transformer
from copy import deepcopy
from threading import Thread
import numpy as np
import json
import torch
import base64
import asyncio
from flcore.optimizer.fedoptimizer import BasedOptimizer
from io import BytesIO

from blockchain.node import Node


class Client():

    def __init__(self, listen_addr, boot_addr, id, model, lr, lr_decay, decay_period, device):
        
        self.device = device
        self.local_module : torch.nn.Module = self.__to_device__(model)
        self.trans = Transformer()
        self.module_length : int = len(self.trans.para_to_list(self.local_module.state_dict(), self.local_module))
        self.learning_rate : float = lr
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.__loss_fn__ : CrossEntropyLoss = self.__to_device__(CrossEntropyLoss())
        self.__optim__ = BasedOptimizer(self.local_module.parameters(), lr)
        self.epoch = 1
        self.lr_decay : float = lr_decay
        self.decay_period = decay_period
        self.valid_loss_min = np.Inf
        self.client_id : int = id
        self.attack : bool = False
        self.server = Node(node_id=str(id))
        self.server_bootstrap(listen_addr, boot_addr)

    
    def server_bootstrap(self, listen_addr : tuple, boot_addr:tuple):
        thread = Thread(target=self._server_bootstrap, args=(listen_addr, boot_addr))
        thread.start()

    def _server_bootstrap(self, listen_addr : tuple, boot_addr : tuple):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.server.listen(listen_addr[1], listen_addr[0]))
        if boot_addr[0] and boot_addr[1]:
            loop.run_until_complete(self.server.node_bootstrap([boot_addr]))
        try:    
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
            loop.stop()

    def set_dataloader(self, dataloader : DataLoader):
        self.train_loader = dataloader[0]
        self.valid_loader = dataloader[1]

    def __to_device__(self, model):
        if self.device == 'cpu':
            return model.cpu()
        elif self.device == 'cuda':
            return model.cuda()
        
    def set_parameters(self, parameters):
        self.local_module.load_state_dict(parameters)
        #self.local_module._set_parameters(parameters)

    def get_parameters(self):
        return self.local_module.state_dict()

    def set_parameters_from_list(self, params_lst):
        '''设置参与方本地模型参数,'''
        if isinstance(params_lst, list) == False:
            raise Exception("模型参数类型不正确.")
        params_ten = self.trans.list_to_para(params_lst, self.local_module)
        self.set_parameters(params_ten)
    

    def label_poison(self, label, a, b):
        a_ten = torch.zeros_like(label, dtype=torch.int64) + a
        b_ten = torch.zeros_like(label, dtype=torch.int64) + b
        temp = torch.zeros_like(label, dtype=torch.int64) + 11
        label = torch.where(label == a, temp, label)
        label = torch.where(label == b, a, label)
        label = torch.where(label == 11, b, label)
        return label
    
    def train_local_model(self):
        self.local_module.train()
        old_weight = deepcopy(self.local_module.state_dict())
        for image, label in self.train_loader:
            #计算输出
            image = self.__to_device__(image)
            label = self.__to_device__(label)
            if self.attack == True:
                if self.client_id == 0 or self.client_id == 1:
                    label = self.label_poison(label, 0, 1)
            output = self.local_module(image)
            #计算损失
            curr_loss = self.__loss_fn__(output, label)
            #初始化梯度参数
            self.__optim__.zero_grad()
            #反向传播
            curr_loss.backward()
            #梯度更新
            self.__optim__.step()
        #学习率衰减
        if self.epoch % self.decay_period == 0:
            self.learning_rate = self.learning_rate * self.lr_decay
        self.epoch += 1
        return (old_weight, self.local_module)
    
    def valid_local_model(self):
        valid_loss = 0.0
        with torch.no_grad():
            self.local_module.eval()  # 验证模型
            accuracy = self.__to_device__(torch.zeros(1))
            for image, label in self.valid_loader:
                image = self.__to_device__(image)
                label = self.__to_device__(label)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.__to_device__(self.local_module(image))
                # calculate the batch loss
                loss = self.__loss_fn__(output, label)
                # update average validation loss 
                valid_loss += loss.item() * image.size(0)
                # convert output probabilities to predicted class(将输出概率转换为预测类)
                pred = output.argmax(1)
                # compare predictions to true label(将预测与真实标签进行比较)
                pred = (pred == label).sum()
                # correct = np.squeeze(correct_tensor.to(device).numpy())
                accuracy += pred.float()
        print("Accuracy:", 100 * accuracy.item() / (len(self.valid_loader) * 32), "%")
        valid_loss = valid_loss / len(self.valid_loader.sampler)

        if valid_loss < self.valid_loss_min:
            self.counter = 0
            print('Validation loss decreased ({:.6f} --> {:.6f}). Learning_rate:{}'.format(self.valid_loss_min, valid_loss, self.learning_rate))
            self.valid_loss_min = valid_loss
        return valid_loss
    
    def send_parameters(self, parameters : dict):
        result = {}
        for key, value in parameters.items():
            bytes_io = BytesIO()
            torch.save(value.cpu(), bytes_io)
            result[key] = base64.b64encode(bytes_io.getvalue()).decode()
        del parameters
        res_str = json.dumps(result)
        del result
        trans = self.server.make_transaction("0", res_str)
        asyncio.run(self.server.broadcast_transaction(trans))


    def update_parameters(self):
        old_weight, _ = self.train_local_model()
        valid_loss = self.valid_local_model()
        new_weight = self.local_module.state_dict()
        with torch.no_grad():
            #Only upload parameters that trainable layer
            self.pseudo_grad = {name : (old_weight[name].data - new_weight[name].data) for name, _ in self.local_module.named_parameters()}
            self.send_parameters(self.pseudo_grad)


