from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

def load_mnist_dataset(train = True)->Dataset: 
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='./data', train=train, transform=transform, download=True)    
    return mnist_dataset

def get_mnist_loader(batch_size = 64)->DataLoader: 
    mnist_dataset = load_mnist_dataset()
    mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True, num_workers=4)
    return mnist_loader

def get_mnist_5_loader(batch_size = 64)->DataLoader: 
    mnist_dataset = load_mnist_dataset(train=False)
    mnist5_dataset = [item for item in mnist_dataset if item[1] == 5]
    mnist5_loader = DataLoader(mnist5_dataset, batch_size=64, shuffle=True, num_workers=4)
    return mnist5_loader

def get_mnist_5_test_set(): 
    mnist_test_set = load_mnist_dataset(train=False)
    mnist5_dataset_test = [item[0] for item in mnist_test_set if item[1] == 5]
    return mnist5_dataset_test