from argparse import ArgumentParser
import os
import sys
import random
import logging
import matplotlib.pyplot as plt
import numpy as np

from scipy import integrate
from sklearn.metrics import mean_squared_error as MSE
import copy

from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate

import torch
from torch.utils.data import DataLoader

from IPython.display import SVG, display
import plotly.graph_objects as go
import plotly.express as px

NUM_POINTS = 0
NUM_EPOCHS = 250
TRAIN_MODEL = True
Z_DIM = 128

def imshow(fig):
    return display(SVG(fig.to_image(format="svg")))

sys.path.append("../src")
from plotter import *
from model import *
from util import *
from data.dataset import SlidingWindowWrapper
#from data.adapter import DataAdapter
from data.synthetic import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def visualize_diff(outputs, targets, portion=1):
    outputs = outputs[:int(len(outputs) * portion)]
    targets = targets[:int(len(targets) * portion)]

    plt.figure(figsize=(14, 10), dpi=180)
    plt.subplot(2, 2, 1)

    n = outputs.shape[0]
    lookahead = outputs.shape[1]

    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 0], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 0], "-o", color="b", label="Actual")
    plt.ylabel('Latitude')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 1], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 1], "-o", color="b", label="Actual")
    plt.ylabel('Longitude')
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 2], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 2], "-o", color="b", label="Actual")
    plt.ylabel('delta_t (hours)')
    plt.legend()
    plt.savefig('result.png')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def eval_loss(model, test_loader, device):
    model.eval()
    loss_total = 0
    sll_meter = AverageMeter()
    tll_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    for index, data in enumerate(test_loader):
        st_x, st_y, _, _, _ = data
        loss, sll, tll = model.loss(st_x, st_y)
        
        loss_meter.update(loss.item())
        sll_meter.update(sll.mean())
        tll_meter.update(tll.mean())
        
    return loss_meter.avg, sll_meter.avg, tll_meter.avg

if __name__ == '__main__':
        
    """The code below is used to set up customized training device on computer"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("You are using GPU acceleration.")
        print("Device name: ", torch.cuda.get_device_name(0))
        print("Number of CUDAs(cores): ", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())

    config = Namespace(hid_dim=128, num_class=369, emb_dim=128, out_dim=0, n_layers=1, 
                   lr=0.0003, momentum=0.9, epochs=NUM_EPOCHS, batch=256, opt='Adam', generate_type=True,
                   read_model=False, seq_len=20, eval_epoch=5,
                   lookahead=1, alpha=0.1, z_dim=Z_DIM, beta=1e-3, dropout=0, num_head=2,
                   nlayers=3, num_points=NUM_POINTS, infer_nstep=10000, infer_limit=13,decoder_n_layer=3,
                   sample=True,constrain_b='sigmoid',b_max=20,s_min=1e-3,clip=1.0)
    
    """
    Prepare logger
    """
    logger = logging.getLogger('full_lookahead{}batch{}'.format(config.lookahead, config.batch))
    logger.setLevel(logging.DEBUG)
    hdlr = logging.FileHandler('full_lookahead{}batch{}.log'.format(config.lookahead, config.batch))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    path = os.path.abspath(os.path.dirname(__file__))
    # Set name of dataset file (add _N to the end of the file name to use a subset of the dataset)
    dataset = np.load(path+'/data/interim/earthquakes_jp.npz', allow_pickle=True)
    print("Dataset size: ", len(dataset.files))
    total = len(dataset.files)
    exclude_from_train = dataset.files[::30]
    val_files = dataset.files[3::30]
    test_files = dataset.files[7::30]
    train_files = list(set(dataset.files).difference(exclude_from_train))

    # Commented out this preprocessing so we can use different dataset subsets
    
    # exclude_from_train = (dataset.files[::30] + dataset.files[1::30] + dataset.files[2::30] + dataset.files[3::30]
    #                     + dataset.files[4::30] + dataset.files[5::30] + dataset.files[6::30] + dataset.files[7::30]
    #                     + dataset.files[8::30] + dataset.files[9::30] + dataset.files[10::30])
    # val_files = dataset.files[3::30]
    # test_files = dataset.files[7::30]
    # train_files = list(set(dataset.files).difference(exclude_from_train))

    # Add own preprocessing
    train_data = [dataset[file] for file in train_files]
    val_data = [dataset[file] for file in val_files]
    test_data = [dataset[file] for file in test_files]
    # Set the first sequence as test to always see the same sequence in calc_lamb
    test_data[0] = dataset[dataset.files[0]]

    trainset = SlidingWindowWrapper(train_data, lookback=20,lookahead=1,normalized=True)
    valset   = SlidingWindowWrapper(val_data, lookback=20,lookahead=1,  normalized=True, min=trainset.min, max=trainset.max)
    testset  = SlidingWindowWrapper(test_data,  lookback=20,lookahead=1,normalized=True, min=trainset.min, max=trainset.max)

    train_loader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
    val_loader   = DataLoader(valset, batch_size=config.batch, shuffle=False)
    test_loader  = DataLoader(testset,batch_size=config.batch, shuffle=False)

    model = DeepSTPP(config, device)
    model_path = path+'/earthquakes'+str(NUM_POINTS)+'.mod'
    if TRAIN_MODEL:
        best_model = train(model, train_loader, val_loader, config, logger, device)
        torch.save(best_model.state_dict(), model_path)
    
    model.load_state_dict(torch.load(model_path))

    T_NSTEP = 100
    X_NSTEP = 100
    Y_NSTEP = 100

    TOTAL_TIME = 35
    print("Last event time: ", test_data[0][-1,0]) # = 29.72
    lambs, x_range, y_range, t_range, his_s, his_t = calc_lamb(model, test_loader, config, device,# scales, biases,
                                                               t_nstep=T_NSTEP, x_nstep=X_NSTEP, y_nstep=Y_NSTEP,
                                                               # xmax=x_max, ymax=y_max, xmin=x_min, ymin=y_min,
                                                               total_time=TOTAL_TIME, round_time=False)


    # transpose all matrices in lambs list otherwise x/y are flipped 
    lambs = [np.transpose(lamb, (1, 0)) for lamb in lambs]

    from plotter import plot_lambst_static, plot_lambst_interactive

    plot_lambst_interactive(lambs, x_range, y_range, t_range, heatmap=True)