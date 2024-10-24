import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from torch import Tensor
import seaborn as sns 


def set_plt_defaults(figsize=(10,10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)

def plot_spiral(X: Tensor,y: Tensor, auto=False, zoom=1, filename:str=None, save_dir=Path(__file__).resolve().parents[2]/"results"/"U1"/"H1"):
    set_plt_defaults()
    X = X.cpu()
    y = y.cpu()
    cm_flare = sns.color_palette("flare", as_cmap=True)
    fig = plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=cm_flare)
    plt.axis('square')
    plt.axis(np.array((-1.1, 1.1, -1.1, 1.1)) * zoom)
    if auto is True: plt.axis('equal')
    plt.axis('off')
    _m, _c = 0, '.15'
    plt.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
    plt.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)
    assert filename is not None
    plt.savefig(fname = save_dir/f"{filename}.png")
    plt.close()

if __name__ == "__main__": 
    
    root_dir = Path(__file__).resolve().parents[2]
    print(root_dir)
