
def train_ancillary_model_epoch(epoch, data_loader, device, models, optimizers, loss_func, scaler, logger):
    total_loss = 0.0
    models["ancillary"].train()
    for batch_idx, (imgs, bboxs), masks in enumerate(data_loader):
        imgs, bboxs, masks = imgs.to(device), bboxs.to(device), masks.to(device).long()
        optimizers["ancillary"].zero_grad()
        outputs = models["ancillary"](imgs, bboxs)

        loss = loss_func(outputs, masks)

        loss.backward()
        optimizers["ancillary"].step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"TRAIN: Epoch:{epoch} at Batch:{batch_idx}/{len(data_loader)} Loss:{loss.item():.3f}")
    
    avg_loss = total_loss/ len(data_loader)
    logger.info(f"ANCILLARY MODEL Epoch:{epoch} average train Loss:{avg_loss:.3f}")
    return avg_loss
