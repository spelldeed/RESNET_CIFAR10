import argparse
import torch
import torch.nn as nn

from utils import load_config, set_seed  
from data import get_dataloaders
from model import get_model
from engine import train_one_epoch, evaluate



def main(config_path):
    config = load_config(config_path)

    set_seed(config["seed"])
    # device = torch.device(config["device"])
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
    train_loader, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)

    # images, labels = next(iter(train_loader))
    # images, labels = images.to(device), labels.to(device)

    # print(f"Input Batch image shape: {images.shape}")
    # print(f"Input Batch label shape: {labels.shape}")

    # outputs =  model(images)
    # print(f"Output batch shape: {outputs.shape}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 2
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("-"*40)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required=True)
    args = parser.parse_args()

    main(args.config)
