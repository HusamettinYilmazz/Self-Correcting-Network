import torch
from utils.eval import compute_confusion_matrix, compute_iou_per_class
from utils.eval import compute_per_class_accuracy

def train_ancillary_model_epoch(epoch, data_loader, device, models, optimizers, loss_func, scaler, logger):
    total_loss = 0.0
    models["ancillary"].train()
    for batch_idx, (imgs, bboxs, masks) in enumerate(data_loader):
        imgs, bboxs, masks = imgs.to(device), bboxs.to(device), masks.to(device).long()
        optimizers["ancillary"].zero_grad()
        outputs = models["ancillary"](imgs, bboxs)

        loss = loss_func(outputs, masks)

        loss.backward()
        optimizers["ancillary"].step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"TRAIN: Epoch:{epoch} at \
                        Batch:{batch_idx}/{len(data_loader)} \
                        Loss:{loss.item():.3f}")
    
    avg_loss = total_loss/ len(data_loader)
    logger.info(f"ANCILLARY MODEL Epoch:{epoch} \
                average train Loss:{avg_loss:.3f}")
    return avg_loss

def validate_ancillary_model(epoch, data_loader, device, models, loss_func, class_names, logger):
    total_loss = 0.0
    all_preds, all_masks = [], []

    models["ancillary"].eval()
    with torch.no_grad():
        for imgs, bboxes, masks in data_loader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device).long()

            preds = models["ancillary"](imgs, bboxes)
            loss = loss_func(preds, masks)
            total_loss += loss.item()

            all_preds.append(preds.argmax(dim=1).cpu())
            all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)

    confusion  = compute_confusion_matrix(all_masks, all_preds, class_names)
    iou        = compute_iou_per_class(confusion)
    acc        = compute_per_class_accuracy(confusion)

    metrics = {
        "avg_loss": total_loss / len(data_loader),
        "mIoU":     iou.mean().item(),
        "avg_acc":  acc.mean().item(),
    }

    logger.info(f"Epoch: {epoch} | Stage 1 validation")
    logger.info(metrics)

    return metrics
