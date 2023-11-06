import torch
import numpy as np


class Transformer():
    def __init__(self) -> None:
        pass

    def para_to_list(self, parameters, module) -> list:
        '''参数构造成一个列表'''
        result_lst = []
        if isinstance(module, torch.nn.Module) == False or isinstance(parameters, dict) == False:
            raise ValueError("模型参数类型不正确")
        for key, value in parameters.items():
            value = value.cpu()
            value = value.reshape([-1])
            lst = value.numpy().tolist()
            result_lst = result_lst + lst
        return result_lst
    
    def grad_to_list(self, parameters, module) -> list:
        '''将梯度构造成一个列表'''
        if isinstance(module, torch.nn.Module) == False:
            raise ValueError("模型参数类型不正确")
        result_lst = []
        for value in parameters:
            ten_grad_peer_para = value.grad.data
            ten_grad_peer_para = ten_grad_peer_para.reshape([-1])
            ten_grad_peer_para = ten_grad_peer_para.cpu()
            lst = ten_grad_peer_para.numpy().tolist()
            result_lst = result_lst + lst
        return result_lst

    
    def list_to_para(self, lst, module):
        '''将一个列表lst构造成模型module的参数dict类型'''
        if isinstance(module, torch.nn.Module) == False:
            raise ValueError("模型参数类型不正确")
        para_dict = module.state_dict()
        for key, value in para_dict.items():
            value = value.cpu()
            field = np.prod(list(value.shape))
            field_lst = lst[:field]
            lst = lst[field:]
            tens = torch.Tensor(field_lst).reshape(value.shape)
            para_dict[key] = tens.cuda()
        return para_dict

    def list_to_grad(self, lst, parameters, module):
        if isinstance(module, torch.nn.Module) == False:
            raise ValueError("模型参数类型不正确")
        with torch.set_grad_enabled(True):
            for value in parameters:
                try:
                    curr_grad = value.grad.data
                except AttributeError as ex:
                    return
                #curr_grad = curr_grad.cpu()
                field = np.prod(list(curr_grad.shape))
                filed_lst = lst[:field]
                lst = lst[field:]
                set_grad = torch.Tensor(filed_lst).reshape(curr_grad.shape)
                value.grad.data = set_grad.cuda()

    def list_add(self, lst1, lst2):
        lst1_ten = torch.tensor(lst1)
        lst2_ten = torch.tensor(lst2)
        if(lst1_ten.shape != lst2_ten.shape):
            raise ValueError("参数形状匹配不正确.")
        return (lst1_ten + lst2_ten).tolist()
    
    def list_divide(self, lst, div_num):
        lst_ten = torch.tensor(lst)
        lst_ten = lst_ten / div_num
        return lst_ten.tolist()