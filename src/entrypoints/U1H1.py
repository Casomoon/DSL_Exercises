
import numpy as np 
import torch
from matplotlib import pyplot as plt 
import random
from ..data.U1H1_spiral import gen_spiral
from ..visualization.U1H1_vis import plot_spiral


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    plot_spiral(X=X, y=y, filename="ground_truth")
