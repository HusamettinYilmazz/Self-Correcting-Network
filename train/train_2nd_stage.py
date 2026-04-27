import torch

def train_correcting_model_epoch(epoch, data_loader, device, models, optimizers, loss_func, scaler, logger):
    total_primary_loss, total_correcting_loss = 0.0, 0.0
    
    models["primary"].train()
    models["correcting"].train()
    models["ancillary"].eval()
    for batch_idx, (imgs, bboxs, masks) in enumerate(data_loader):
        imgs, bboxs, masks = imgs.to(device), bboxs.to(device), masks.to(device).long()
        
        primary_outputs = models["primary"](imgs)
        with torch.no_grad():
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
            logger.info(f"TRAIN PRIMARY MODEL: Epoch:{epoch} at Batch:{batch_idx}/{len(data_loader)} Loss:{primary_loss.item():.3f}")
            logger.info(f"TRAIN CORRECTING NETWORK: Epoch:{epoch} at Batch:{batch_idx}/{len(data_loader)} Loss:{correcting_loss.item():.3f}")
    
    primary_avg_loss = total_primary_loss/ len(data_loader)
    correcting_avg_loss = total_correcting_loss/ len(data_loader)
    logger.info(f"PRIMARY MODEL Epoch:{epoch} average train Loss:{primary_avg_loss:.3f}")
    logger.info(f"CORRECTING NETWORK Epoch:{epoch} average train Loss:{correcting_avg_loss:.3f}")
    return primary_avg_loss, correcting_avg_loss
