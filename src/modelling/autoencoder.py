import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE_Dense(nn.Module):
    def __init__(self, input_size: tuple[int, int], first_layer_dims = 256, layers = 5) -> "AE_Dense":
        super(AE_Dense, self).__init__() 
        self.input_size = input_size
        out_dims = first_layer_dims
        self.encoder = nn.Sequential(   # sequential chains multiple layers after one another
            nn.Linear(input_size[0]*input_size[1], out_dims) # input images have 28x28 pixels -> input layer requires 28x28 large input layer
        )
        for i in range(layers):
            self.encoder.append(nn.ReLU())
            in_dims = out_dims 
            out_dims = int(first_layer_dims/ (2**(i+1)))
            self.encoder.append(nn.Linear(in_dims, out_dims))


        latent_dims  =out_dims
        self.decoder = nn.Sequential()
        prev_layer_out = latent_dims
        for _ in range(layers): 
            in_dims = prev_layer_out
            out_dims = in_dims*2
            self.decoder.append(nn.Linear(in_dims, out_dims))
            self.decoder.append(nn.ReLU())
            prev_layer_out = out_dims
        self.decoder.append(nn.Linear(prev_layer_out, input_size[0]*input_size[1]))
        self.decoder.append(nn.Sigmoid()) # output layer function


    def forward(self,x: torch.Tensor):
        x = x.view((x.size(0),-1)) # reshape from (bs, 1, 28, 28) to (bs, 784)
        latent = self.encoder(x)
        pred = self.decoder(latent)
        # gpto1 used here, couldnt resolve (torch.Size([64, 1, 28, 28])) that is different to the input size (torch.Size([64, 784])) is deprecated
        pred = pred.view(x.size(0), 1, self.input_size[0], self.input_size[1])
        return pred
    
class AE_Conv(nn.Module):
    def __init__(self, input_channels: tuple[int, int])-> "AE_Conv":
        super(AE_Conv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, kernel_size=32, kernel_size=3, stride= 2, padding = 1)
        )
        self.decoder = nn.Sequential() 

def train_step(model, batch_X, loss_fn: torch.nn.Module, opti: torch.optim.Optimizer): 
    opti.zero_grad()
    outputs = model(batch_X)
    loss: torch.Tensor = loss_fn(outputs, batch_X)
    # backward pass
    loss.backward()
    # param update
    opti.step()
    # loss calculation for this epoch, loss this epoch * batch_size
    loss = loss.item()*batch_X.size(0)
    return loss

def fit(model: nn.Module, loss_fn, train_loader: torch.Tensor, epochs: int, optimizer: torch.optim.Optimizer=None, logger = None):
    model.to(device)
    model.train()
    for epoch in range(epochs): 
        epoch_running_loss = 0.0
        total_samples  = 0
        for batch_X, _ in train_loader: 
            #logger.info(batch_X.min())
            #logger.info(batch_X.max())
            assert batch_X.min()>=0 
            assert batch_X.max()<=1
            batch_X = batch_X.to(device)
            batch_loss = train_step(model, batch_X, loss_fn, optimizer)
            epoch_running_loss += batch_loss
            total_samples += batch_X.size(0)
        epoch_loss = epoch_running_loss/total_samples
        logger.info(f"Epoch [{epoch+1}/{epochs}] - BinaryCrossentropyLoss: {epoch_loss:.4f}")
    return model