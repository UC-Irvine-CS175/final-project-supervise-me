"""Prototype for PyTorch model for BPS data

    - Mode collapse sometimes occurs when training the model
    - Model is not able to learn the data
    - Model is not able to generalize to the validation set over 3 epochs
    - Model loss is not decreasing over epochs

    - Accuracy and loss have an inverse relationship across runs
    
    Shoutout to Jacob Campbell!!

"""
import os
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))
from sklearn.metrics import accuracy_score
from src.dataset.bps_dataset import BPSMouseDataset, BPSDataModule
from torchmetrics import Accuracy
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    VFlipBPS,
    HFlipBPS,
    RotateBPS,
    RandomCropBPS,
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
    save_dir:           str = root / 'models' / 'SAP_model'
    batch_size:         int = 16
    max_epochs:         int = 10
    accelerator:        str = 'auto'
    devices:            int = 1

class LeNet5(nn.Module):
    """
    LeNet-5 architecture for BPS classification
    """
    def __init__(self):
        super().__init__()
        """LeNet-5 architecture 

        Only the first layer, the first fully connected layer, 
        and the output layer are modified to fit the BPS dataset.
        """
        # Lets begin with an example of a 1x200x200 image
        # Input channels is 1, output channels is 6, kernel size is 5
        # Strides default to 1
        # Kernel size is 5x5, so the weighted sum is 25 pixels
        # output_width = ((input_width - kernel_size + (2 * padding))/ stride) + 1
        # output_height = ((input_height - kernel_size + (2 * padding))/ stride) + 1
        # output_depth = K (number of filters, user-specified)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # output_width = ((200 - 5)/2) + 1 = 98
        # output_height = ((200 - 5)/2) + 1 = 98
        # output_depth = 6
        # Conv2d outputs a 6x196x196 matrix
        # The 196x196 is a result of the kernel size of 5x5 with 6 channels and stride of 1
        # With 210 weight parameters

        # Pool reduces spatial dimensions by 2,
        # based on parameters stride = 2 and kernel_size= 2
        # Requires no additional arguments
        # Preserves the number of channels
        # ((size - kernel)/ stride) + 1
        # ((196 - 2)/2) + 1 = 98
        # Resulting tensor is 6x98x98
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Kernel size is 5x5, so the weighted sum is 25 pixels
        # output_width = ((input_width - kernel_size + (2 * padding))/ stride) + 1
        # output_height = ((input_height - kernel_size + (2 * padding))/ stride) + 1
        # output_depth = K (number of filters, user-specified)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 47 * 47, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class BPSClassifier(pl.LightningModule):
    """
    Classifier for BPS dataset
    """
    def __init__(self):
        super().__init__()
        self.model = LeNet5()
        self.val_acc = Accuracy(task='binary',
                                num_classes=2,
                                multidim_average='global')

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('train_loss', loss)            # Tensorboard
        wandb.log({'train_loss' : loss})        # Weights and Biases
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        y_truth = torch.argmax(y, dim=1)

        # Accuracy is the average of the number of an entire batch of correct predictions
        val_acc = torch.mean((torch.eq(y_pred, y_truth)).float())
        # self.log('val_loss', val_loss)          # Tensorboard
        wandb.log({'val_loss' : val_loss, 'val_acc' : val_acc})      # Weights and Biases

    def test_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer



def main():
    """
    Testing function for BPS AutoEncoder

        1) Define configuration options
        2) Define training dataset
        3) Define training dataloader
        4) Define validation dataset
        5) Define validation dataloader
        6) Define model
        7) Define trainer
        8) Train model
        9) Save model
        10) Test model
        11) Save model

    Notes:
        - Loss function should decrease with each epoch
        - Validation loss should be lower than training loss
        - Test loss should be lower than validation loss
    """
    # Define configuration options
    config = BPSConfig()

    # Define training dataset
    train_dataset = BPSMouseDataset(config.train_meta_fname,
                                    config.data_dir,
                                    transform=transforms.Compose([
                                        NormalizeBPS(),
                                        ResizeBPS(224, 224),
                                        VFlipBPS(),
                                        HFlipBPS(),
                                        RotateBPS(90),
                                        RandomCropBPS(200, 200),
                                        ToTensor()]),
                                    file_on_prem=True)

    # Define training dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=False, num_workers=12)


    # Define validation dataset
    validate_dataset = BPSMouseDataset(config.val_meta_fname,
                                       config.data_dir,
                                       transform=transforms.Compose([
                                            NormalizeBPS(),
                                            ResizeBPS(224, 224),
                                            VFlipBPS(),
                                            HFlipBPS(),
                                            RotateBPS(90),
                                            RandomCropBPS(200, 200),
                                            ToTensor()]),
                                        file_on_prem=True)

    # Define validation dataloader
    validate_dataloader = DataLoader(validate_dataset, batch_size=config.batch_size,
                                     shuffle=False, num_workers=12)

    # Initialize wandb logger
    wandb.init(
    # set the wandb project where this run will be logged
    project="SAP-lnet-from-scratch",
    dir=config.save_dir,
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0003,
    "architecture": "LeNET",
    "dataset": "BPS Microscopy Mouse Dataset",
    "epochs": config.max_epochs,})

    # model
    autoencoder = BPSClassifier()

    # train model with training and validation dataloaders
    trainer = pl.Trainer(default_root_dir=config.save_dir,
                         accelerator=config.accelerator,
                         devices=config.devices,
                         max_epochs=config.max_epochs,
                         profiler="simple")

    trainer.fit(model=autoencoder,
                train_dataloaders=train_loader,
                val_dataloaders=validate_dataloader)

    # test model
    # Automate saving checkpoints from training with this assignment
    trainer = pl.Trainer(default_root_dir=config.save_dir,
                         accelerator=config.accelerator,
                         devices=config.devices,
                         max_epochs=config.max_epochs)
    
    # # Load checkpoint from training
    # model = BPSAutoEncoder.load_from_checkpoint(config.save_dir + 'lightning_logs/version_0/checkpoints/epoch=9.ckpt')

    # Predict with the model
    # assign an image to x

    # # Define test dataset

    # # Define test dataloader

    

if __name__ == "__main__":
    main()
