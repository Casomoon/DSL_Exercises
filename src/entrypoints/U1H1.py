
import numpy as np 
import torch
from matplotlib import pyplot as plt 
import random
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from ..data.U1H1_spiral import gen_spiral, train_test
from ..visualization.U1H1_vis import plot_spiral, plot_spiral_model
from ..modelling.linear_pytorch import SimpleLinear, LinearWithActivation, fit
from ..log import Log

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
proj_root = Path(__file__).resolve().parents[2]
loc_log = proj_root/"logs"
if not loc_log.exists(): 
    loc_log.mkdir()
logger = Log("u1h1", "INFO", loc_log).getLogger()
# dataset specifications
N = 10000
D = 2
C = 3 

def set_seeds(seed : int):
    random.seed(seed)
    torch.manual_seed(seed) 
    
def set_plt_defaults(figsize=(10,10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)

def main():
    set_seeds(42)
    set_plt_defaults()
    X,y = gen_spiral(N,D,C)
    # wrap it into dataloader objs
    batch_size = 64
    train, test = train_test(X,y)
    # linear model first 
    plot_spiral(X=X, y=y, filename="ground_truth")
    #train loop
    linear = SimpleLinear(C) 
    linear_relu = LinearWithActivation(C)
    loss_fn = nn.CrossEntropyLoss()
    opti = optim.SGD(linear_relu.parameters(), lr = 0.01)
    
    trained_linear  = fit(linear, loss_fn, train, epochs = 5, optimizer=opti, logger=logger)
    plot_spiral_model(test_dataloader=test, model=trained_linear, filename="trained_linear")
    # fit linear model with rectified linear unit
    trained_linear_relu = fit(linear_relu, loss_fn, train, epochs = 200, optimizer=opti, logger = logger)
    plot_spiral_model(test_dataloader=test, model=trained_linear_relu, filename="trained_linear_relu" )


