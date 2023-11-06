import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

def __to_cuda__(module):
    if(torch.cuda.is_available()):
        return module.cuda()

def draw_loss_weight(data_paths):
    fig = plt.figure(figsize=(8.5, 3.7))
    gs = gridspec.GridSpec(1, 2)
    plt.subplots_adjust(left=0.092, right=0.98, bottom=0.15, top=0.9, wspace=0.25)

    i=0
    for data_path in data_paths:
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#DCDCDC")
        plt.grid(color="#FFFFFF")

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        data = pd.read_csv(data_path)
        x = data["weight"]
        y = data["loss"]
        if i == 1:
            y = [d - 0.02 for d in y]
        plt.tick_params(labelsize=13)
        linespace = np.linspace(min(y), max(y), 5)
        ax.set_xlabel("Aggregation Weight", fontsize=15)
        ax.set_ylabel("Training Loss", fontsize=15)
        ax.set_yticks(linespace)
        if i == 0:
            ax.set_yticklabels(["%.1f"%l for l in np.linspace(16.7, 17.1, 5)])
        else:
            ax.set_yticklabels(["%.1f"%l for l in np.linspace(16.8, 17.2, 5)])

        base_line = [y[0] for i in y]
        critical_line1 = [x[0] for i in x]
        critical_line2 = [0.103 for i in x]
        plt.plot(x, y, label="loss decent")
        plt.plot(x, base_line, label="base line")
        #plt.plot(critical_line1, y, ":k", label="critical point")
        #plt.plot([x[0]], [y[0]], "mx")
        if i == 0:
            #plt.plot(critical_line2, y, ":k")
            #plt.plot([0.102], [y[0]], "mx")
            pass

        plt.rcParams.update({"font.size": 15})
        plt.legend(loc="upper left")
        i += 1
    plt.savefig(r"C:\Users\10270\Desktop\paper\paper\mypaper\data\test.tiff", dpi=300, format="tiff")

    plt.show()

def draw_worth(data_paths):
    fig = plt.figure(figsize=(8.5, 3.7))

    gs = gridspec.GridSpec(1, 2)
    i = 0
    for data_path in data_paths:
        ax = fig.add_subplot(gs[0, i])
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.98, wspace=0.15)

        ax.tick_params(labelsize=13)
        ax.set_ylim(-0.2, 1.2)
        ax.set_ylabel("Worthiness", fontsize=15)
        ax.set_xlabel("Update Quality", fontsize=15)
        ax.set_yticks(np.linspace(0, 1, 2))

        data = pd.read_csv(data_path)

        x = data["loss_decent"]
        y = data["value"]
        plt.plot(x, y)
        i += 1

    plt.savefig(r"C:\Users\10270\Desktop\paper\paper\mypaper\data\Fig 4.tiff", dpi=300, format="tiff")
    plt.show()

def draw_figure(ax, datas, title):
    ax.set_facecolor("#DCDCDC")
    ax.grid(color="#FFFFFF")
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(labelsize=11)

    ax.set_xlabel("Federated learning round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_xticks(np.linspace(0, 50, 6))
    ax.set_yticks(np.linspace(30, 90, 4))
    #ax.set_title(title, fontsize=13)

    for path_label in datas:
        print(path_label)
        data = pd.read_csv(path_label[0])
        x = data["step"]
        accuracy = data["accuracy"]
        accuracy = [i * 100 for i in accuracy]
        ax.plot(x, accuracy, label=path_label[1])
    

def draw_all_figure(data_path_list):
    fig = plt.figure(figsize=(4.5, 3.7))

    i = 0
    n = 3
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.9, wspace=0)
    for path in data_path_list:
        ax = fig.add_subplot(gs[0, i])
        draw_figure(ax, path, "{} IID + {} non-IID".format(10- n, n))
        if i == 0:
            plt.rcParams.update({"font.size": 13})
            ax.legend(loc="lower right")
        i += 1
        n += 2


    #plt.savefig(r"C:\Users\10270\Desktop\paper\paper\mypaper\data\test.tiff", dpi=300, format="tiff")

    plt.show()

def draw_weight(data_path):
    fig = plt.figure(figsize=(5, 3.7))
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.9, wspace=0.25)
    for i in [1, 2, 3, 4]:
        ax = fig.add_subplot(220 + i)
        ax.set_facecolor("#DCDCDC")
        plt.grid(color="#FFFFFF",zorder=0)
        plt.tick_params(labelsize=11)
        if i in [1, 3]:
            ax.set_ylabel("Weight", fontsize=13)
        if i in [3, 4]:
            ax.set_xlabel("Trainer", fontsize=13)
        
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(range(10))
        ax.set_ylim(0, 0.5)
        ax.set_yticks([0, 0.25, 0.5])

        data = pd.read_csv(data_path)
        index_name = ["weight{}".format(i) for i in range(10)]
        line_data = [data[index][i-1] for index in index_name]
        plt.bar([x for x in range(10)], line_data, zorder=5)
    plt.savefig(r"C:\Users\10270\Desktop\paper\paper\mypaper\data\Fig 9.2.tiff", dpi=300, format="tiff")
    plt.show()

def draw_DP_bar(data_path):
    fig = plt.figure(figsize=(8.5, 3.7))
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.95, wspace=0.15)
    line = [0, 1]
    gs = gridspec.GridSpec(1, 2)
    for i in line:
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#DCDCDC")
        plt.grid(color="#FFFFFF", zorder=0)
        plt.tick_params(labelsize=11)
        ax.set_xlabel("Federated learning round", fontsize=13)
        ax.set_ylabel("Test Accuracy (%)", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks(np.linspace(0, 80, 5))

        width = 2.5
        data = pd.read_csv(data_path)
        raw_data = data["raw_data"]
        fix_data = data["fix_data=1"]
        dynamic_data = data["dynamic_data=1"]
        step = [10, 20, 30, 40, 50]

        e = 0.1 if i == 0 else 1
        ax.bar([s - width for s in step], raw_data, width, label="Without", zorder=5)
        ax.bar([s + (0) for s in step], fix_data, width, label="Fix (ε={})".format(e), zorder=5)
        ax.bar([s + width for s in step], dynamic_data, width, label="Dynamic(ε={})".format(e), color="brown", zorder=5)

        ax.set_ylim(0, 0.8)
        ax.set_yticklabels([0, 20, 40, 60, 80])
        ax.set_xticks(step)

        plt.rcParams.update({"font.size": 13})
        plt.legend(loc="lower right")

    plt.savefig(r"C:\Users\10270\Desktop\paper\paper\mypaper\data\Fig 8.tiff", dpi=300, format="tiff")
    plt.show()

def draw_weight_total(pic_path):
    fig = plt.figure(figsize=(4.5, 3.7))
    plt.subplots_adjust(left=0.135, right=0.98, bottom=0.15, top=0.9, wspace=0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#DCDCDC")
    plt.grid(color="#FFFFFF")

    ax.set_ylabel("Normalized Weight", fontsize=13)
    ax.set_xlabel("Federated learning round", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tick_params(labelsize=11)
    ax.set_yticks(np.linspace(1, 2.2, 5))

    data = pd.read_csv(pic_path)
    x = data["step"]
    y = data["weight_total"]

    plt.plot(x, y, label="ALW")

    plt.savefig(r"C:\Users\10270\Desktop\paper\paper\mypaper\data\Fig 9.3.tiff", dpi=300, format="tiff")
    plt.show()


def draw_consensus_delay(pic_path):
    fig = plt.figure(figsize=(8.5, 3.7))
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.98, wspace=0.21)
    gs = gridspec.GridSpec(1, 2)
    line = [0, 1]
    node_size_list = [(3, 5), (4, 6)]
    for i in line:
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#DCDCDC")
        plt.grid(color="#FFFFFF")

        ax.set_ylabel("Consensus Deley (ms)", fontsize=15)
        ax.set_xlabel("Number of Nodes", fontsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tick_params(labelsize=13)
        ax.set_yticks([200, 300, 400, 500])

        data = pd.read_csv(pic_path)
        x = data["num"]
        ax.set_xticks(x)
        our_delay = data["our={}".format(node_size_list[i][0])]
        gossip_delay = data["gossip={}".format(node_size_list[i][1])]

        plt.plot(x, our_delay, "-D", label="our")
        plt.plot(x, gossip_delay, "-D", label="gossip")

        plt.rcParams.update({"font.size": 15})
        plt.legend(loc="lower right")
    plt.savefig(r"C:\Users\10270\Desktop\paper\paper\mypaper\data\Fig 10.tiff", dpi=300, format="tiff")
    plt.show()

def __to_cuda__(module):
    if(torch.cuda.is_available()):
        return module.cuda()



if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    pic_path = ".\\data\\"
    draw_all_figure([
        [(pic_path+"data2.csv", "ALW"), (pic_path+"data1.csv", "FedAvg")],
    ])
    #draw_loss_weight([pic_path + "loss decent -6.csv", pic_path + "loss decent -50.csv"])
    #draw_loss_weight([pic_path + "non-iid=1,epoch=5.csv", pic_path + "non-iid=2,epoch=5.csv"])
    #draw_worth([pic_path + "loss decent and fault c=5.csv", pic_path + "loss decent and fault c=8.csv"])
    #draw_weight(pic_path+"weights.csv")
    #draw_DP_bar(pic_path+"privacy efficiency.csv")
    #draw_weight_total(pic_path+"weight_total.csv")
    #draw_consensus_delay(pic_path+"consensus_delay.csv")