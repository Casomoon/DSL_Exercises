from ..data.U1H2_MNIST import get_mnist_loader
from ..modelling.autoencoder import AE_Dense, AE_Conv fit
from ..log import Log
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

proj_root = Path(__file__).resolve().parents[2]
loc_log = proj_root/"logs"
if not loc_log.exists(): 
    loc_log.mkdir()
logger = Log("u1h2", "INFO", loc_log).getLogger()

def main():
    mnist_train = get_mnist_loader()
    ae_dense = AE_Dense(input_size = (28,28), first_layer_dims=256, layers = 5)
    ae_dense_trained = fit(ae_dense, 
                           loss_fn = nn.BCELoss(), # !!!! changed normalization from -1/1 to 0/1 range so i can use BCE
                           train_loader= mnist_train, 
                           epochs = 20, 
                           optimizer= optim.Adam(ae_dense.parameters()),
                           logger = logger)
    ae_conv = A