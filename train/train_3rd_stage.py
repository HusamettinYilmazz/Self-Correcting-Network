from itertools import cycle
import torch

from utils.eval import compute_confusion_matrix, compute_iou_per_class
from utils.eval import compute_per_class_accuracy, plot_confusion_matrix

def unsupervised_loss(w_primary_outputs, correcting_logits):

    log_probs = torch.log_softmax(w_primary_outputs, dim=1)
    pseudo = torch.softmax(correcting_logits, dim=1).detach()

    unsup_loss = -(pseudo * log_probs).sum(dim=1).mean()
    
    return unsup_loss

def compute_lambda(epoch):
    return min(1.0, epoch/10)

def train_primary_model_epoch(epoch, data_loaders, device, models,
                       optimizers, loss_func, scaler, logger):
    
    total_loss = 0.0
    
    models["primary"].train()
    models["correcting"].eval()
    models["ancillary"].eval()
    for batch_idx, ((f_imgs, f_bbox, f_masks), (w_imgs, w_bbox)) in \
        enumerate(zip(data_loaders["f_loader"], cycle(data_loaders["w_loader"]))):
        f_imgs, f_masks = f_imgs.to(device), f_masks.to(device).long()
        w_imgs, w_bbox = w_imgs.to(device), w_bbox.to(device)

        f_primary_outputs = models["primary"](f_imgs)
        w_primary_outputs = models["primary"](w_imgs)

        with torch.no_grad():
                ancillary_outputs = models["ancillary"](w_imgs, w_bbox)
                correcting_logits = models["correcting"](
                    w_primary_outputs.detach(), 
                    ancillary_outputs.detach()
                )

        primary_loss = loss_func(f_primary_outputs, f_masks)
        unsup_loss = unsupervised_loss(w_primary_outputs, correcting_logits)
        lambda_u = compute_lambda(epoch)

        loss = primary_loss + lambda_u * unsup_loss
        optimizers["primary"].zero_grad()
        loss.backward()
        optimizers["primary"].step()
        
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"TRAIN PRIMARY MODEL: Epoch:{epoch} \
                        at Batch:{batch_idx}/{len(data_loaders['f_loader'])} \
                        Primary Loss:{primary_loss.item():.3f} |\
                        Unsupervised Loss:{unsup_loss.item():.3f} &\
                        Combined Loss:{loss.item():.3f}")
    
    avg_loss = total_loss/ len(data_loaders["f_loader"])
    logger.info(f"PRIMARY MODEL Epoch:{epoch} average train Loss:{avg_loss:.3f}")
    
    return avg_loss

def validate_primary_model(epoch, data_loader, device, model, loss_func,
                     class_names, logger, save_dir=None):
    
    model["primary"].eval()

    total_loss = 0.0
    total_cm = None

    with torch.no_grad():
        for imgs, _, masks in data_loader:
            imgs = imgs.to(device)
            masks = masks.to(device).long()


            outputs = model(imgs)

            loss = loss_func(outputs, masks)
            total_loss += loss.item()

            # preds = outputs.argmax(dim=1)

            cm = compute_confusion_matrix(
                masks,
                outputs,
                class_names,
                ignore_index=255
            )

            total_cm = cm if total_cm is None else total_cm + cm

    iou_per_class = compute_iou_per_class(total_cm)
    acc_per_class = compute_per_class_accuracy(total_cm)

    metrics = {
        "avg_loss": total_loss / len(data_loader),
        "iou_per_class": iou_per_class,
        "acc_per_class": acc_per_class,
    }

    logger.info(f"Epoch: {epoch} | Stage 3 validation")
    logger.info(metrics)

    if save_dir is not None:
        plot_confusion_matrix(total_cm, class_names, save_path=save_dir)

    return metrics
