import argparse
import torch

from utils import load_config, set_seed  
from data import get_dataloaders
from model import get_model



def main(config_path):
    config = load_config(config_path)

    set_seed(config["seed"])
    device = torch.device(config["device"])
    train_loader, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    print(f"Input Batch image shape: {images.shape}")
    print(f"Input Batch label shape: {labels.shape}")

    outputs =  model(images)
    print(f"Output batch shape: {outputs.shape}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = "str", required=True)
    args = parser.parse_args()

    main(args.config)
