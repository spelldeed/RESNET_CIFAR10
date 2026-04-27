import argparse
import logging
import torch
import torch.nn as nn

from utils import load_config, setup_logging
from data import get_dataloaders
from model import get_model
from engine import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    config = load_config(args.config)

    device_cfg = config["training"].get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_cfg == "auto" else torch.device(device_cfg)

    _, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    logger.info(f"Checkpoint : {args.checkpoint}")
    logger.info(f"Test loss  : {test_loss:.4f}")
    logger.info(f"Test acc   : {test_acc:.4f}")
