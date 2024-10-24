import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from torch import Tensor
import seaborn as sns 
import torch 
from torch.utils.data import DataLoader


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

def plot_spiral_model(test_dataloader: DataLoader, model, filename):
    model.cpu()
    all_X = []
    all_y = []
    
    for batch_X, batch_y in test_dataloader:
        all_X.append(batch_X)
        all_y.append(batch_y)
    
    # Concatenate all batches into one tensor
    full_X = torch.cat(all_X, dim=0)
    full_y = torch.cat(all_y, dim=0)
    
    mesh = np.arange(-1.1, 1.1, 0.01)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plot_spiral(full_X, full_y, filename=filename)

if __name__ == "__main__": 
    
    root_dir = Path(__file__).resolve().parents[2]
    print(root_dir)
