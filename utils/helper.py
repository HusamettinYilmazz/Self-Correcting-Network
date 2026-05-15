import os 
import yaml
import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
class Config:
    def __init__(self, config_dict):
        self.experiment = config_dict.get("experiment", {})
        self.data = config_dict.get("data", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})

    def __reper__(self):
        return f"Config(experiment={self.experiment}, data={self.data} model={self.model}, training={self.training})"
     

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config

def save_checkpoint(epoch, model, optimizer, cur_lr, val_acc, config, 
                    train_transform, val_transform, save_dir, model_name="model"):
    
    checkpoint_path = os.path.join(save_dir, f'epoch{epoch}_{model_name}.pth')

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': cur_lr,
            'val_accuracy': val_acc,
            'config': config,
            'train_transform': train_transform,
            'val_transform': val_transform

        }, checkpoint_path)

    print(f"Epoch:{epoch} checkpoint has been saved at:{checkpoint_path}")

def lr_vs_epoch(num_epochs, lrs, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), lrs, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs. Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'epoch_vs_lr.png'), bbox_inches='tight')

def get_boundaries(mask):
    # mask: [B, H, W]
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)

    mask = mask.unsqueeze(1).float()  # [B,1,H,W]

    dilated = F.conv2d(mask, kernel, padding=1)
    eroded  = -F.conv2d(-mask, kernel, padding=1)

    boundary = (dilated - eroded) > 0
    return boundary.squeeze(1)

def boundary_f1(pred, target):
    pred_b = get_boundaries(pred)
    target_b = get_boundaries(target)

    tp = (pred_b & target_b).sum().float()
    fp = (pred_b & ~target_b).sum().float()
    fn = (~pred_b & target_b).sum().float()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)

    bf1 = 2 * precision * recall / (precision + recall + 1e-6)
    return bf1.item()

def class_distribution(preds, num_classes):
    # preds: [B,H,W]
    hist = torch.zeros(num_classes, device=preds.device)

    for c in range(num_classes):
        hist[c] = (preds == c).sum()

    return hist / hist.sum()

def imbalance_indicator(preds, targets, num_classes):
    pred_dist = class_distribution(preds, num_classes)
    gt_dist   = class_distribution(targets, num_classes)

    return (pred_dist - gt_dist).abs().mean().item()
