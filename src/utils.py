
import numpy as np
import torch
import yaml 
import random 
import os
import time
import csv

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
    os.makedirs(run_dir, exist_ok = True)
    return run_dir

def save_config(config, run_dir):
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)


def log_metrics(run_dir, epoch, train_loss, train_acc, test_loss, test_acc):
    file_path = os.path.joins(run_dir, "metrics.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        
        writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

def save_model(model, path):
    torch.save(model.state_dict(), path)

