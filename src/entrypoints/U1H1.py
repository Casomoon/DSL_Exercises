
import numpy as np 
import torch
from matplotlib import pyplot as plt 
from ..data.U1H1_spiral import gen_spiral


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset specifications
N = 10000
D = 2
C = 3 

def set_plt_defaults(figsize=(10,10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)

def main(): 
    set_plt_defaults()
    gen_spiral()
