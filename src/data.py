import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_dataloaders(config):
    root = config['dataset']['root']
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    if config.get("debug", {}).get("small_subset", False):
        train_dataset = Subset(train_dataset, range(200))
        test_dataset  = Subset(test_dataset,  range(100))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


