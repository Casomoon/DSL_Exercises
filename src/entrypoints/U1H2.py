from ..data.U1H2_MNIST import get_mnist_loader, get_mnist_5_loader
from ..modelling.autoencoder import AE_Dense, AE_Conv, fit
from ..log import Log
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import torch 
import numpy as np

proj_root = Path(__file__).resolve().parents[2]
loc_log = proj_root/"logs"
if not loc_log.exists(): 
    loc_log.mkdir()
logger = Log("u1h2", "INFO", loc_log).getLogger()
drop_test_imgs = proj_root/"results"/"U1"/"H2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    mnist_train = get_mnist_loader()
    ae_dense = AE_Dense(input_size = (28,28), first_layer_dims=256, layers = 5)
    ae_dense_trained = fit( ae_dense, 
                            loss_fn = nn.BCELoss(), # !!!! changed normalization from -1/1 to 0/1 range so i can use BCE
                            train_loader= mnist_train, 
                            epochs = 20, 
                            optimizer= optim.Adam(ae_dense.parameters()),
                            logger = logger)
    ae_conv = AE_Conv(input_channels=1)# 1 input channel not 3 rgb
    ae_conv_trained  =fit(  ae_conv,
                            loss_fn = nn.BCELoss(), # !!!! changed normalization from -1/1 to 0/1 range so i can use BCE
                            train_loader= mnist_train, 
                            epochs = 20, 
                            optimizer= optim.Adam(ae_conv.parameters()),
                            logger = logger
                            )
    five_test_images = get_mnist_5_loader()
    dataiter = iter(five_test_images)
    images = next(dataiter)[0] # get one batch
    images = images.to(device)
    ae_conv_trained.to(device)
    ae_conv_trained.eval()

    # reconstruction
    with torch.no_grad():
        ae_conv_trained.to(device)
        ae_conv_trained.eval()
        reconstructed_conv = ae_conv_trained(images)
        ae_dense_trained.to(device)
        ae_dense_trained.eval()
        reconstructed_dense = ae_dense_trained(images)

    reconstructed_conv_np = reconstructed_conv.cpu().numpy()
    reconstructed_dense_np = reconstructed_dense.cpu().numpy()    

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

    for i in range(5):
        ax = axes[0, i]
        ax.imshow(np.squeeze(reconstructed_dense_np[i]), cmap='gray')
        ax.axis('off')
        ax.set_title('Reconstructed')
    for i in range(5):
        ax = axes[1, i]
        ax.imshow(np.squeeze(reconstructed_conv_np[i]), cmap='gray') # squeeze remove singleton dim 
        ax.axis('off')                                          # 1 28 28 -> 28 28         
        ax.set_title('Reconstructed')

    plt.savefig(drop_test_imgs/"plot_rec.png")
    plt.close()

    
                          