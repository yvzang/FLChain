from copyreg import pickle
from flcore.server import serverbase, serverALW, serverfedavg
import torchvision
from flcore.client import clientbase, clientFedAvg
import torch
from ResNet import ResNet18
from resnet18 import VGG, Net, build_vgg11, FedAvgCNN
from copy import deepcopy
import argparse
from utils.readData import *


def read_dataloader(split_num, batch_size, hete_num=0, dataset_type="cifar10", valid_size=0.1):
    dataloaders = []
    datasets = split_dataset(split_num, dataset_type)
    weight_list = [False for i in range(split_num-hete_num)]
    for w in range(hete_num):
        weight_list.append(True)
    data_weight = zip(datasets, weight_list)
    for dataset, weight in data_weight:
        dataloaders.append(read_non_iid_dataset(dataset_type, dataset, weight, batch_size, valid_size))
    return dataloaders

def create_clients(client_type, algorithm, dataloaders, model, lr, lr_decay, decay_period, device):
    clients = []
    for i in range(len(dataloaders)):
        c = client_type(i, deepcopy(model), lr, lr_decay, decay_period, device)
        c.set_dataloader(dataloaders[i])
        clients.append(c)

    return clients

    
def run(args : argparse.Namespace):
    #create model
    model = args.model
    if model in 'cnn':
        if args.dataset in 'mnist':
            model = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
        elif args.dataset in 'cifar10':
            model = FedAvgCNN(in_features=3, num_classes=10, dim=1600)
    elif model in 'resnet18':
        if args.dataset in 'mnist':
            model = ResNet18(in_channels=1)
        elif args.dataset in 'cifar10':
            model = ResNet18(in_channels=3)
    #split dataloader
    dataloaders = read_dataloader(args.client_num, args.batch_size, args.hete_num, args.dataset, args.valid_size)
    #read test dataloader
    testloader = read_test_dataset(args.dataset, args.batch_size)
    #create server
    if args.agg_algorithm in 'FedAvg':
        clients = create_clients(clientFedAvg.ClientFedAvg, args.agg_algorithm, dataloaders, model, args.learning_rate, args.lr_decay, args.lr_decay_period, args.device)
        ser = serverfedavg.ServerFedAvg(model, args.device, testloader, args.batch_size, clients, args.test_epoch, args.train_round, args.data_save_path, args.agg_rate)
    elif args.agg_algorithm in 'Single':
        clients = create_clients(clientbase.Client, args.agg_algorithm, dataloaders, model, args.learning_rate, args.lr_decay, args.lr_decay_period, args.device)
        ser = serverbase.ServerBase(model, args.device, testloader, args.batch_size, clients, args.test_epoch, args.train_round, args.data_save_path, args.agg_rate)
    elif args.agg_algorithm in 'ALW':
        clients = create_clients(clientbase.Client, args.agg_algorithm, dataloaders, model, args.learning_rate, args.lr_decay, args.lr_decay_period, args.device)
        proxy_loader = read_proxy_dataset(args.dataset, args.batch_size)
        ser = serverALW.ServerALW(model, args.device, testloader, proxy_loader, args.batch_size, clients, args.test_epoch, args.train_round, args.data_save_path, args.agg_rate)
    ser.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"], help="trained dataset. [mnist, cifar10]")
    parser.add_argument('-cn', "--client_num", type=int, default=10, help="clients number.")
    parser.add_argument('-niid', "--non_iid_number", type=int, default=0, help="non-iid dataset number.")
    parser.add_argument('-bs', "--batch_size", type=int, default=8, help="batch size.")
    parser.add_argument('-vs', "--valid_size", type=float, default=0.1, help="valid dataset size.")
    parser.add_argument('-htn', "--hete_num", type=int, default=0, help="number of heterogeneity dataset")
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.005, help="client learning rate.")
    parser.add_argument('-ld', "--lr_decay", type=int, default=1, help="learning rate decay rate.")
    parser.add_argument('-dp', "--lr_decay_period", type=int, default=1, help="learning rate decay period. How many training round decay lr.")
    parser.add_argument('-m', "--model", type=str, default='resnet18', help="trained modle. [cnn, resnet18]")
    parser.add_argument('-algo', "--agg_algorithm", type=str, default='ALW', help="aggregation algorithm. [FedAvg, Single, ALW]")
    parser.add_argument('-dv', "--device", type=str, default='cuda', choices=["cuda", "cpu"], help="training device. [cpu, cuda]")
    parser.add_argument('-te', "--test_epoch", type=int, default=5, help="how many epoch test loss and accuracy.")
    parser.add_argument('-tr', "--train_round", type=int, default=50, help="total training rounds.")
    parser.add_argument('-pth', "--data_save_path", type=str, default="./data/data2.csv", help="data of result save path.")
    parser.add_argument('-ar', "--agg_rate", type=float, default=1, help="aggregation percentage of all clients.")
    args = parser.parse_args()
    run(args)


