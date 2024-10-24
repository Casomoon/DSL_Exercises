import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
def gen_spiral(num_samples_per_class: int = 1000, dimensions: int = 2, num_classes: int = 3): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.zeros(num_samples_per_class*num_classes, dimensions).to(device)
    y = torch.zeros(num_samples_per_class*num_classes, dtype=torch.long).to(device)
    for current_class_idx in range(num_classes): 
        index = 0
        # generate N evenly spaced points between 0 and 1, normalized parameter that runs for each class, 
        # used as facotr for calculating the spiral radius
        t = torch.linspace(start = 0, end = 1, steps = num_samples_per_class)
        # inner var is used as arg for sincos, depicts the radians
        angles = torch.linspace( 
            start = (2*math.pi/num_classes) * (current_class_idx),  #start angle for the current class
            end   = (2*math.pi/num_classes) * (2+current_class_idx),#+2 steps for the end angle of the class -> spiral loops as t progresses
            steps = num_samples_per_class
        ) + torch.randn(num_samples_per_class)*0.2
        for ix in range(num_samples_per_class*current_class_idx, num_samples_per_class*(current_class_idx+1)):
            X[ix] = t[index] * torch.FloatTensor((
                torch.sin(angles[index]), torch.cos(angles[index])
                ))
            y[ix] = current_class_idx
            index +=1
    print(f" X has shape {X.size()}, y has shape {y.size()}")
    return X,y

def train_test(X: torch.Tensor,y: torch.Tensor, test_split:int = 0.2, bs = 64)-> tuple[DataLoader]: 
    assert X.device == y.device
    dataset = TensorDataset(X,y)
    num_samples = len(dataset)
    num_test_samples = int(num_samples * test_split) 
    num_train_samples = num_samples -num_test_samples
    train_set, test_set = random_split(dataset, [num_train_samples, num_test_samples]) 
    train_loader = DataLoader(train_set, batch_size=64, shuffle= True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle= True)
    return train_loader, test_loader
    