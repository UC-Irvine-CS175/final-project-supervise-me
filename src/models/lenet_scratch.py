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
from src.dataset.bps_dataset import BPSMouseDataset
from src.dataset.bps_dataset_multi_label import BPSMouseDataset as BPSMouseMultiLabel
from src.dataset.bps_datamodule import BPSDataModule

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
    num_workers:        int = 1
    bucket_name:        str = "nasa-bps-training-data"
    s3_path:            str = "Microscopy/train"
    s3_client:          str = boto3.client('s3', config=Config(signature_version=UNSIGNED))

class LeNet5(nn.Module):
    """
    LeNet-5 architecture for BPS classification
    """
    def __init__(self, 
                 num_labels: int):
        super().__init__()
        """
        LeNet-5 architecture 
        
        Args:
            num_labels (int): Specifies the unique number of labels, equal to the output of the final fully connected layer.

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
        self.fc3 = nn.Linear(84, num_labels) #6 is the num of different labels

        

    def forward(self, x: torch.Tensor):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class BPSClassifier(pl.LightningModule):
    def __init__(self, 
                 learning_rate: float = 3e-4, 
                 num_labels: int = 2):
        """
        Classifier for BPS dataset

        Args:
            learning_rate (float): Specifies the learning rate of the model
            num_labels (int): Specifies how many different labels there are
        """
        super().__init__()
        self.model = LeNet5(num_labels)
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.val_acc = Accuracy(task='binary',
                                num_classes=2,
                                multidim_average='global')
        self.learning_rate = learning_rate
        

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = torch.argmax(y, dim=1) # Fe: [1, 0]-> 0, X-ray: [0, 1] -> 1
        loss = F.cross_entropy(y_hat, y)
        wandb.log({'train_loss' : loss})        # Weights and Biases
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_truth = torch.argmax(y, dim=1)
        y_pred = torch.argmax(y_hat, dim=1)
        val_loss = F.cross_entropy(y_hat, y)
        # Accuracy is the average of the number of an entire batch of correct predictions
        val_acc = torch.mean((torch.eq(y_pred, y_truth)).float())
        wandb.log({'val_loss' : val_loss, 'val_acc' : val_acc})      # Weights and Biases

    def test_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# A main function for our original dataset, that returns the particle-type label
# e.g. Fe or X-ray
def main_original_dataset():
    """
    Testing function for BPS AutoEncoder that predicts single-labels

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
    my_settings = BPSConfig()

    wandb.init(
    # set the wandb project where this run will be logged
    project="SAP-lnet-from-scratch",
    dir=my_settings.save_dir
    )
    
    
    # Define datamodule
    bps_datamodule = BPSDataModule(train_csv_file=my_settings.train_meta_fname,
                                   train_dir=my_settings.data_dir,
                                   val_csv_file=my_settings.val_meta_fname,
                                   val_dir=my_settings.data_dir,
                                   batch_size=wandb.config.batch_size,
                                   num_workers=my_settings.num_workers,
                                   s3_client= my_settings.s3_client,
                                   bucket_name= my_settings.bucket_name,
                                   s3_path= my_settings.s3_path,
                                   multi_label= False)
    
    ##### UNCOMMENT THE LINE BELOW TO DOWNLOAD DATA FROM S3!!! #####
    #bps_datamodule.prepare_data()
    ##### WHEN YOU ARE DONE REMEMBER TO COMMENT THE LINE ABOVE TO AVOID
    ##### DOWNLOADING THE DATA AGAIN!!! #####
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage='train')
    bps_datamodule.setup(stage='validate')

    train_loader = bps_datamodule.train_dataloader()
    val_loader = bps_datamodule.val_dataloader()

    # model
    autoencoder = BPSClassifier(learning_rate=wandb.config.lr, num_labels=2)

    # train model with training and validation dataloaders
    trainer = pl.Trainer(default_root_dir=my_settings.save_dir,
                         accelerator=my_settings.accelerator,
                         devices=my_settings.devices,
                         max_epochs=wandb.config.epochs,
                         profiler="simple")

    trainer.fit(model=autoencoder,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # test model
    # Automate saving checkpoints from training with this assignment
    trainer = pl.Trainer(default_root_dir=my_settings.save_dir,
                         accelerator=my_settings.accelerator,
                         devices=my_settings.devices,
                         max_epochs=wandb.config.epochs)
    
    # # Load checkpoint from training
    # model = BPSAutoEncoder.load_from_checkpoint(config.save_dir + 'lightning_logs/version_0/checkpoints/epoch=9.ckpt')

    # Predict with the model
    # assign an image to x

    # # Define test dataset

    # # Define test dataloader
    wandb.finish()

# A main function for our multi-label dataset
# e.g. (particle-type, dosage)
def main_multi_label_dataset():
    """
    Testing function for BPS AutoEncoder that predicts multi-labels

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
    my_settings = BPSConfig()

    wandb.init(
    # set the wandb project where this run will be logged
    project="SAP-lnet-from-scratch",
    dir=my_settings.save_dir
    )
    
    
    # Define datamodule
    bps_datamodule = BPSDataModule(train_csv_file=my_settings.train_meta_fname,
                                   train_dir=my_settings.data_dir,
                                   val_csv_file=my_settings.val_meta_fname,
                                   val_dir=my_settings.data_dir,
                                   batch_size=wandb.config.batch_size,
                                   num_workers=my_settings.num_workers,
                                   s3_client= my_settings.s3_client,
                                   bucket_name= my_settings.bucket_name,
                                   s3_path= my_settings.s3_path,
                                   multi_label= True)
    
    ##### UNCOMMENT THE LINE BELOW TO DOWNLOAD DATA FROM S3!!! #####
    # bps_datamodule.prepare_data()
    ##### WHEN YOU ARE DONE REMEMBER TO COMMENT THE LINE ABOVE TO AVOID
    ##### DOWNLOADING THE DATA AGAIN!!! #####
    
    # Using BPSDataModule's setup, define the stage name ('train' or 'val')
    bps_datamodule.setup(stage='train')
    bps_datamodule.setup(stage='validate')

    train_loader = bps_datamodule.train_dataloader()
    val_loader = bps_datamodule.val_dataloader()

    # model
    autoencoder = BPSClassifier(learning_rate=wandb.config.lr, num_labels=6)

    # train model with training and validation dataloaders
    trainer = pl.Trainer(default_root_dir=my_settings.save_dir,
                         accelerator=my_settings.accelerator,
                         devices=my_settings.devices,
                         max_epochs=wandb.config.epochs,
                         profiler="simple")

    trainer.fit(model=autoencoder,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # test model
    # Automate saving checkpoints from training with this assignment
    trainer = pl.Trainer(default_root_dir=my_settings.save_dir,
                         accelerator=my_settings.accelerator,
                         devices=my_settings.devices,
                         max_epochs=wandb.config.epochs)
    
    # # Load checkpoint from training
    # model = BPSAutoEncoder.load_from_checkpoint(config.save_dir + 'lightning_logs/version_0/checkpoints/epoch=9.ckpt')

    # Predict with the model
    # assign an image to x

    # # Define test dataset

    # # Define test dataloader
    wandb.finish()
    

if __name__ == "__main__":
    sweep_config = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'train_loss'
            },
        'parameters': {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 0.0005}
        }
    }
        

    #starting the sweep
    sweep_id = wandb.sweep(
            sweep=sweep_config,
            project="SAP-lnet-from-scratch"
        )

    # wandb.agent(sweep_id=sweep_id, function=main_original_dataset, count=10)
    wandb.agent(sweep_id=sweep_id, function=main_multi_label_dataset, count=10)
