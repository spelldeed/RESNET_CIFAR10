import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


def get_dataloaders(config):
    root = config['dataset']['root']
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']

    
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root = root, train = True, download = True, transform = transform)
    test_dataset = torchvision.datasets.CIFAR10(root = root, train = False, download = True, transform = transform)

    if config.get("debug", {}).get("small_subset", False):
        train_dataset = Subset(train_dataset, range(200))
        test_dataset = Subset(test_dataset, range(100))
        
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


