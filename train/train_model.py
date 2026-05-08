import os
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import Model
from datasets.pascal_voc import VOCDataset
from utils import Config, load_config, lr_vs_epoch, save_checkpoint
from utils.logger import Logger

from models.primary_model import PrimarySegmentationModel
from models.ancillary_model import AncillarySegmentationModel
from models.self_correcting_model import SelfCorrectingNetwrokFactory





def train(config: Config, checkpoint_path=None):

    dataset_path = os.path.join(ROOT, config.data['dataset_path'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.Resize(512, 512),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.5),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    train_val_dataset_path = os.path.join(dataset_path, 
                                        config.data['train_dataset_path'])
    
    fully_sup_train_dataset = VOCDataset(data_path= train_val_dataset_path,
                                    data_type="train",
                                    is_sup= True,
                                    transform=train_transform)
    generator = torch.Generator().manual_seed(42)
    f1_dataset, f2_dataset = random_split(fully_sup_train_dataset, 
                                          [len(fully_sup_train_dataset)//2, ...],
                                            generator=generator)
    
    weak_train_dataset = VOCDataset(data_path= train_val_dataset_path,
                                    data_type="train",
                                    is_sup= False,
                                    transform=train_transform)
    
    val_dataset = VOCDataset(data_path= train_val_dataset_path,
                               data_type="val",
                               is_sup= True,
                               transform=val_transform)
    
    fully_sup_train_loader = DataLoader(dataset=fully_sup_train_dataset, 
                                        batch_size=config.training['batch_size'],
                                        shuffle= True, pin_memory= True)
    
    f1_loader = DataLoader(dataset=f1_dataset, 
                           batch_size=config.training['batch_size'],
                           shuffle= True, pin_memory= True)
    
    f2_loader = DataLoader(dataset=f2_dataset, 
                           batch_size=config.training['batch_size'],
                           shuffle= True, pin_memory= True)
    
    weak_train_loader = DataLoader(dataset=weak_train_dataset, 
                                   batch_size=config.training['batch_size'],
                                   shuffle= True, pin_memory= True)
    train_loaders = {
        "f_loader": fully_sup_train_loader,
        "w_loader": weak_train_loader,
        "f1_loader": f1_loader,
        "f2_loader": f2_loader
    }
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=config.training['batch_size'], 
                            shuffle= True, pin_memory= True)

    primary_model = PrimarySegmentationModel(
        num_classes=config.model["num_classes"]).to(device)
    
    ancillary_model = AncillarySegmentationModel(
        num_classes=config.model["num_classes"]).to(device)
    
    correcting_model = SelfCorrectingNetwrokFactory().build_correction_module(
        variant = "conv_correction", 
        num_classes=config.model["num_classes"]
    ).to(device)

    models = {
        "primary": primary_model,
        "ancillary": ancillary_model,
        "correcting": correcting_model
    }

    optimizers = {
        "primary": AdamW(primary_model.parameters(), 
                         lr=config.training['learning_rate'],
                         weight_decay=float(config.training['weight_decay'])
        ),

        "ancillary": AdamW(ancillary_model.parameters(), 
                           lr=config.training['learning_rate'],
                           weight_decay=float(config.training['weight_decay'])
        ),

        "correcting": AdamW(correcting_model.parameters(), 
                            lr=config.training['learning_rate'],
                            weight_decay=float(config.training['weight_decay'])
        ),
    }

    schedulers = {
        "primary": lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers["primary"], mode='min', factor=0.1, patience=2
        ),

        "ancillary": lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers["ancillary"], mode='min', factor=0.1, patience=2
        ),

        "correcting": lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers["correcting"], mode='min', factor=0.1, patience=2
        )
    }

    loss_func = nn.CrossEntropyLoss(ignore_index=255)
    scaler = GradScaler()

    """
    # if checkpoint_path:
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     starting_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     optimizer.param_groups[0]['lr'] = checkpoint['learning_rate']

    # else:
    #     starting_epoch = 1
    """

    starting_epoch = 1
    save_dir = os.path.join(ROOT, config.data['output_path'], config.experiment['name'], config.experiment['version'])
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(save_dir)
    logger.info(f"Starting the experiment: {config.experiment['name']} {config.experiment['version']}")
    logger.info(f"Using device: {device}")
    logger.info(f"Fully Supervised Training dataset size: {len(fully_sup_train_dataset)}")
    logger.info(f"Weak Training dataset size: {len(weak_train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    lrs = []
    logger.info(f"Starting training from epoch: {starting_epoch}")
    
    stage1_training_loop(
        starting_epoch, config, train_loaders, val_loader, train_transform,
        val_transform, device, models, optimizers, schedulers, loss_func, logger, save_dir
    )



    ## Stage 2
    stage2_training_loop(
        starting_epoch, config, train_loaders, val_loader, train_transform, 
        val_transform, device, models, optimizers, schedulers, loss_func, logger, save_dir
    )

    ## stage 3
    stage3_training_loop(
        starting_epoch, config, train_loaders, val_loader, train_transform, 
        val_transform,device, models, optimizers, schedulers, loss_func, logger, save_dir
    )
    
    ## log that training is done successfully.
    logger.info("All the 3 stages are finished successfully")


if __name__ == "__main__":
    config = load_config(os.path.join(ROOT, "config/config.yml"))
    train(config)

"""

DONE:
    1. Build the 3 models
    2. Initilize 2 optimizers
    3. Be sure about input consistency of (data_loaders, models, optimizers) across the 3 stages
    4. Before you build f_training_loader split f to f1 and f2 and 
        build a data_loader instance for each
    5. Build 3 function: one for each stage training loop
    6. optimizer, schedular etc parsing to the above 3 functions
    7. Write configuration yaml file in configs/

what to do next:

    8. Run on Kaggle
"""