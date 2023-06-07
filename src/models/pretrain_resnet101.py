"""
Prototype for PyTorch model for BPS data
"""
import os
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))
from sklearn.metrics import accuracy_score
from src.dataset.bps_dataset import BPSMouseDataset
from src.dataset.bps_datamodule import BPSDataModule
from torchmetrics import Accuracy
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    ToThreeChannels,
    ToTensor
)
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from dataclasses import dataclass
from datetime import datetime
import io
from io import BytesIO
from PIL import Image
import numpy as np
import random

@dataclass
class BPSConfig:
    """ Configuration options for BPS Microscopy dataset.

    Args:
        data_dir: Path to the directory containing the image dataset. Defaults
            to the `data/processed` directory from the project root.

        train_meta_fname: Name of the training CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_train.csv'

        val_meta_fname: Name of the validation CSV file.
            Defaults to 'meta_dose_hi_hr_4_post_exposure_test.csv'
        
        save_dir: Path to the directory where the model will be saved. Defaults
            to the `models/SAP_model` directory from the project root.

        batch_size: Number of images per batch. Defaults to 4.

        max_epochs: Maximum number of epochs to train the model. Defaults to 3.

        accelerator: Type of accelerator to use for training.
            Can be 'cpu', 'gpu', 'tpu', 'ipu', 'auto', or None. Defaults to 'auto'
            Pytorch Lightning will automatically select the best accelerator if
            'auto' is selected.

        devices: Number of devices to use for training. Defaults to 1.
    """
    data_dir:           str = root / 'data' / 'processed'
    train_meta_fname:   str = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    val_meta_fname:     str = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    save_dir:           str = root / 'models' / 'RESNET_101'
    batch_size:         int = 16
    max_epochs:         int = 1
    accelerator:        str = 'auto'
    devices:            int = 1
    num_workers:        int = 12
    dm_stage:           str = 'train'

class BPSClassifier(pl.LightningModule):
    """
    Classifier for BPS dataset
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        # Convert OHE to class indices
        y = torch.argmax(y, dim=1) # Fe: [1, 0]-> 0, X-ray: [0, 1] -> 1
        loss = F.cross_entropy(y_hat, y)

        wandb.log({'train_loss' : loss})        # Weights and Biases
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        # y = torch.tensor([self.label_dict[i] for i in y])
        
        # Convert OHE to class indices
        y = torch.argmax(y, dim=1)

        val_loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        
        # Accuracy is the average of the number of an entire batch of correct predictions
        val_acc = torch.mean((torch.eq(y_pred, y)).float())

        wandb.log({'val_loss' : val_loss, 'val_acc' : val_acc})      # Weights and Biases

    def test_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer



def main():
    config = BPSConfig()

    # Tip 1: fix random seed
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Instantiate BPSDataModule
    bps_datamodule = BPSDataModule(train_csv_file=config.train_meta_fname,
                                   train_dir=config.data_dir,
                                   val_csv_file=config.val_meta_fname,
                                   val_dir=config.data_dir,
                                   resize_dims=(224, 224),
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers)
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage=config.dm_stage)
    bps_datamodule.setup(stage='validate')
    
    # Initialize wandb logger
    wandb.init(
    # set the wandb project where this run will be logged
    project="resnet-101-pretrain",
    dir=config.save_dir,
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0003,
    "architecture": "ResNet101",
    "dataset": "BPS Microscopy Mouse Dataset",
    "epochs": config.max_epochs,})

    model = torch.hub.load("pytorch/vision",
                            "resnet101",
                            weights="IMAGENET1K_V2")
    
    # Replace the 1000 classification layer with a binary classification layer
    # This is because the pre-trained model was trained on 1000 classes (ImageNet)
    # But we only have 2 classes (Fe and X-ray) in our data subset
    # TLDR: fc1000 -> fc2
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Freeze pre-trained layers except for the last fully connected layer 
    # This was done to speed up testing & debugging the training step
    # The model still makes use of the pre-trained weights in the frozen layers
    # but the gradients will not be updated for these layers
    # Your team can experiment with fine-tuning in various ways, e.g.
        # Unfreezing more layers
        # Adding more layers
        # Tuning hyperparameters
    # See http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # Here is a visual guide to the layers in ResNet101 (hover over the layers for more info):
    # http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50

    trainer = pl.Trainer(default_root_dir=config.save_dir,
                         accelerator=config.accelerator,
                         devices=config.devices,
                         max_epochs=config.max_epochs,
                         profiler="simple")
    trainer.fit(model= BPSClassifier(model),
                train_dataloaders=bps_datamodule.train_dataloader(),
                val_dataloaders=bps_datamodule.val_dataloader())
    wandb.finish()    

sweep_config = {
    'method': 'grid',
    'name': 'sweep',
    'run_cap': 1,
    'metric': {
        'goal': 'minimize', 
        'name': 'train_loss'
        },
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        # 'lr': {'max': 0.1, 'min': 0.0001}
     }
    }
    

    #starting the sweep
sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="resnet-101-pretrain",
    )

wandb.agent(sweep_id = sweep_id, function=main, count=10)


if __name__ == '__main__':
    main()