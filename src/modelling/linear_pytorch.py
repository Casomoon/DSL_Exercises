import torch.nn as nn 
import torch
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
class SimpleLinear(nn.Module):
    def __init__ (self, num_output_classes):
        super(SimpleLinear,self).__init__()
        self.lin1 = nn.Linear(2,num_output_classes)
    def forward(self,x): 
        return self.lin1(x)

class LinearWithActivation(nn.Module): 
    def __init__(self, num_output_classes):
        super(LinearWithActivation, self).__init__()
        self.lin_hidden=nn.Linear(2,64)
        self.lin_out = nn.Linear(64,num_output_classes)
        self.relu = nn.ReLU()
    def forward(self,x): 
        x=self.relu(self.lin_hidden(x))
        x=self.lin_out(x)
        return x    


# chatgpt(o1) used for train_step and fit method, never used conventional pytorch, always Lightning
def train_step(model, batch_X, batch_y, loss_fn: torch.nn.Module, opti: torch.optim.Optimizer): 
    
    opti.zero_grad()
    outputs = model(batch_X)
    loss: torch.Tensor = loss_fn(outputs, batch_y)
    # backward pass
    loss.backward()
    # param update
    opti.step()
    # loss calculation for this epoch, loss this epoch * batch_size
    loss = loss.item()*batch_X.size(0)
    predicted = torch.argmax(outputs, dim = 1)
    correct= (predicted == batch_y).sum().item()
    return loss, correct
     

def fit(model: nn.Module, loss_fn, train_loader: torch.Tensor, epochs: int, optimizer: torch.optim.Optimizer=None, logger = None):
    model.to(device)
    model.train()
    for epoch in range(epochs): 
        epoch_running_loss = 0.0
        correct_preds = 0 
        total_samples  = 0
        for batch_X, batch_y in train_loader: 
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_loss, batch_correct = train_step(model, batch_X, batch_y, loss_fn, optimizer)
            epoch_running_loss += batch_loss
            correct_preds+= batch_correct
            total_samples += batch_y.size(0)
        epoch_loss = epoch_running_loss/total_samples
        epoch_acc = correct_preds/total_samples
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.2f}%")
    return model



        