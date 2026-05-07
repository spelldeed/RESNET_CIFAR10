# import argparse
# import torch
# import torch.nn as nn

# from utils import load_config
# from data import get_dataloaders
# from model import get_model
# from engine import evaluate


# def run_evaluation(config_path, checkpoint_path):
#     config = load_config(config_path)

#     device_cfg = config["training"].get("device", "auto")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
#         if device_cfg == "auto" else torch.device(device_cfg)

#     _, test_loader = get_dataloaders(config)
#     model = get_model(config).to(device)

#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(
#         checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
#     )

#     criterion = nn.CrossEntropyLoss()
#     test_loss, test_acc = evaluate(model, test_loader, criterion, device)

#     print(f"Checkpoint: {checkpoint_path}")
#     print(f"Test loss : {test_loss:.4f}")
#     print(f"Test acc  : {test_acc:.4f}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--checkpoint", type=str, required=True)
#     args = parser.parse_args()

#     run_evaluation(args.config, args.checkpoint)