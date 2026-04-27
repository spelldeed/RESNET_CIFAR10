
import numpy as np
import torch
import yaml
import random
import os
import time
import csv
import logging


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_dir(config):
    exp_name = config["exp_name"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", exp_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(run_dir):
    log_path = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )


def save_config(config, run_dir):
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)


def log_metrics(run_dir, epoch, train_loss, train_acc, test_loss, test_acc):
    file_path = os.path.join(run_dir, "metrics.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

        writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_acc":        best_acc,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint["scheduler_state"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint["epoch"], checkpoint["best_acc"]
