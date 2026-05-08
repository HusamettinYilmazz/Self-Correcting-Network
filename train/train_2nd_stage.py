import torch

from utils.eval import compute_confusion_matrix, compute_iou_per_class
from utils.eval import compute_per_class_accuracy, plot_confusion_matrix

def train_correction_model_epoch(epoch, data_loader, device, models, optimizers, loss_func, scaler, logger):
    total_primary_loss, total_correcting_loss = 0.0, 0.0
    
    models["primary"].train()
    models["correcting"].train()
    models["ancillary"].eval()
    for batch_idx, (imgs, bboxs, masks) in enumerate(data_loader):
        imgs, bboxs, masks = imgs.to(device), bboxs.to(device), masks.to(device).long()
        
        primary_outputs = models["primary"](imgs)
        with torch.no_grad():
            models["ancillary"].freeze()
            ancillary_outputs = models["ancillary"](imgs, bboxs)

        correcting_outputs = models["correcting"](
            primary_outputs.detach(), 
            ancillary_outputs.detach()
        )

        primary_loss = loss_func(primary_outputs, masks)
        correcting_loss = loss_func(correcting_outputs, masks)

        optimizers["primary"].zero_grad()
        primary_loss.backward()
        optimizers["primary"].step()

        optimizers["correcting"].zero_grad()
        correcting_loss.backward()
        optimizers["correcting"].step()
        
        total_primary_loss += primary_loss.item()
        total_correcting_loss += correcting_loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"TRAIN PRIMARY MODEL: Epoch:{epoch} \
                        at Batch:{batch_idx}/{len(data_loader)} \
                        Loss:{primary_loss.item():.3f}")
            
            logger.info(f"TRAIN CORRECTING NETWORK: Epoch:{epoch} \
                        at Batch:{batch_idx}/{len(data_loader)} \
                        Loss:{correcting_loss.item():.3f}")
    
    primary_avg_loss = total_primary_loss/ len(data_loader)
    correcting_avg_loss = total_correcting_loss/ len(data_loader)
    logger.info(f"PRIMARY MODEL Epoch:{epoch} average train Loss:{primary_avg_loss:.3f}")
    logger.info(f"CORRECTING NETWORK Epoch:{epoch} average train Loss:{correcting_avg_loss:.3f}")
    logger.info(f"Epoch:{epoch} average train Loss:{(primary_avg_loss+correcting_avg_loss)/2:.3f}")
    return primary_avg_loss, correcting_avg_loss

def validate_correction_model(epoch, data_loader, device, models, loss_func, class_names, logger, save_dir=None):
    total_primary_loss, total_correcting_loss = 0.0, 0.0
    primary_total_cm, correcting_total_cm = None, None

    models["primary"].eval()
    models["correcting"].eval()
    models["ancillary"].eval()
    with torch.no_grad():
        for imgs, bboxes, masks in data_loader:
            imgs = imgs.to(device)
            masks = masks.to(device).long()

            primary_outputs = models["primary"](imgs)
            ancillary_outputs = models["ancillary"](imgs, bboxes)
            correcting_outputs = models["correcting"](primary_outputs, ancillary_outputs)
            
            primary_loss = loss_func(primary_outputs, masks)
            correcting_loss = loss_func(correcting_outputs, masks)

            total_primary_loss += primary_loss.item()
            total_correcting_loss += correcting_loss.item()

            primary_preds = primary_outputs.argmax(dim=1)
            correcting_preds = correcting_outputs.argmax(dim=1)
            
            primary_cm = compute_confusion_matrix(
                masks,
                primary_preds,
                class_names,
                ignore_index=255
            )

            correcting_cm = compute_confusion_matrix(
                masks,
                correcting_preds,
                class_names,
                ignore_index=255
            )

            primary_total_cm = primary_cm if primary_total_cm is None \
                                                else primary_total_cm + primary_cm
            
            correcting_total_cm = correcting_cm if correcting_total_cm is None \
                                                else correcting_total_cm + correcting_cm

    primary_iou_per_class = compute_iou_per_class(primary_total_cm)
    primary_acc_per_class = compute_per_class_accuracy(primary_total_cm)

    correcting_iou_per_class = compute_iou_per_class(correcting_total_cm)
    correcting_acc_per_class = compute_per_class_accuracy(correcting_total_cm)

    metrics = {
        "primary_avg_loss": total_primary_loss / len(data_loader),
        "correcting_avg_loss": total_correcting_loss / len(data_loader),
        
        "primary_iou_per_class": primary_iou_per_class,
        "primary_acc_per_class": primary_acc_per_class,

        "correcting_iou_per_class": correcting_iou_per_class,
        "correcting_acc_per_class": correcting_acc_per_class
    }

    logger.info(f"Epoch: {epoch} | Stage 2 validation")
    logger.info(metrics)

    if save_dir is not None:
        plot_confusion_matrix(primary_total_cm, class_names, save_path=save_dir)
        plot_confusion_matrix(correcting_total_cm, class_names, save_path=save_dir)

    return metrics
