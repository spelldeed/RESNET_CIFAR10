import argparse
import logging
import torch
import torch.nn as nn
import os

from utils import (
    load_config, set_seed, create_run_dir, save_config,
    setup_logging, log_metrics, save_checkpoint, load_checkpoint,
)
from data import get_dataloaders
from model import get_model
from engine import train_one_epoch, evaluate

logger = logging.getLogger(__name__)


def build_optimizer(config, model):
    opt = config["optimizer"]
    return torch.optim.SGD(
        model.parameters(),
        lr=opt["lr"],
        momentum=opt["momentum"],
        weight_decay=opt["weight_decay"],
    )


def build_scheduler(config, optimizer, epochs):
    name = config["scheduler"]["name"]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    return None


def main(config_path, resume_path=None):
    config = load_config(config_path)
    run_dir = create_run_dir(config)
    save_config(config, run_dir)
    setup_logging(run_dir)

    set_seed(config["seed"])

    device_cfg = config["training"].get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_cfg == "auto" else torch.device(device_cfg)
    logger.info(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)

    epochs    = config["training"]["epochs"]
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, epochs)

    start_epoch = 1
    best_acc    = 0.0

    if resume_path is not None:
        start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch += 1
        logger.info(f"Resumed from {resume_path} | start epoch {start_epoch} | best acc so far {best_acc:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        test_loss, test_acc   = evaluate(model, test_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Test loss {test_loss:.4f} acc {test_acc:.4f} | "
            f"LR {current_lr:.6f}"
        )

        log_metrics(run_dir, epoch, train_loss, train_acc, test_loss, test_acc)

        if config["logging"].get("save_last", True):
            save_checkpoint(
                os.path.join(run_dir, "model_last.pth"),
                epoch, model, optimizer, scheduler, best_acc,
            )

        if test_acc > best_acc:
            best_acc = test_acc
            if config["logging"].get("save_best", True):
                save_checkpoint(
                    os.path.join(run_dir, "model_best.pth"),
                    epoch, model, optimizer, scheduler, best_acc,
                )

    logger.info(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(args.config, args.resume)
