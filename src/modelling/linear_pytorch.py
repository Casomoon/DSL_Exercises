import torch.nn as nn 
class SimpleLinear(nn.Module):
    def __init__ (self, ):
        super(SimpleLinear,self)
        self.lin1 = nn.Linear(2,3)