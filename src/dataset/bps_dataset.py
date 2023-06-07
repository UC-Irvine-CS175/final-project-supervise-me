"""
This module contains the BPSMouseDataset class which is a subclass of torch.utils.data.Dataset.
"""
import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms, utils
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
import pyprojroot
from pyprojroot import here

root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
pytorch_dataset_dir = root / 'dataset'
data_dir = root / 'data'

import sys
sys.path.append(str(root))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import io
from io import BytesIO
from PIL import Image
from src.dataset.augmentation import (
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)

from src.data_utils import get_bytesio_from_s3

class BPSMouseDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for the BPS microscopy data.

    args:
        meta_csv_file (str): name of the metadata csv file
        meta_root_dir (str): path to the metadata csv file
        s3_client (boto3.client): boto3 client for s3
        bucket_name (str): name of bucket from AWS open source registry.
        transform (callable, optional): Optional transform to be applied on a sample.
        file_on_prem (bool): True if the data is on the local file system, False if the data is on S3

    attributes:
        s3_client (boto3.client): boto3 client for s3
        bucket_name (str): name of bucket from AWS open source registry.
        on_prem (bool): True if the data is on the local file system, False if the data is on S3
        meta_dir (str): path to the metadata csv file
        meta_csv (str): name of the metadata csv file
        meta_df (pd.DataFrame): dataframe containing the metadata
        transform (callable): The transform to be applied on a sample.

    raises:
        ValueError: if the metadata csv file does not exist
    """

    def __init__(
            self,
            meta_csv_file:str,
            meta_root_dir:str,
            s3_client: boto3.client = None,
            bucket_name: str = None,
            transform=None,
            file_on_prem:bool = True):
        """
        Constructor for BPSMouseDataset class.
        """    
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.on_prem = file_on_prem
        self.meta_dir = meta_root_dir
        self.meta_csv = meta_csv_file

        meta_csv_full_path = f'{self.meta_dir}/{self.meta_csv}'

        if not file_on_prem:
            # create a BytesIO object from the file contents
            file_buffer = get_bytesio_from_s3(
                s3_client=s3_client, bucket_name=bucket_name, file_path=meta_csv_full_path
            )
            self.meta_df = pd.read_csv(file_buffer)
        else:
            self.meta_df = pd.read_csv(meta_csv_full_path)
        
        # One-hot encode the particle type for classification task
        self.meta_df = pd.get_dummies(self.meta_df, columns=["particle_type"])

        self.transform = transform
    
    def __len__(self):
        """
        Returns the number of images in the dataset.

        returns:
          len (int): number of images in the dataset
        """
        return len(self.meta_df)

    def __getitem__(self, idx):
        """
        Fetches the image and corresponding label for a given index.

        Args:
            idx (int): index of the image to fetch

        Returns:
            img_tensor (torch.Tensor): tensor of image data
            label (int): label of image
        """

        # get the bps image file name from the metadata dataframe at the given index
        row = self.meta_df.iloc[idx]
        img_fname = row["filename"]

        # formulate path to image given the root directory (note meta.csv is in the
        # same directory as the images)

        img_key = f"{self.meta_dir}/{img_fname}"

        # write code to fetch image from s3 bucket or from the local file system based on
        # the boolean value of self.on_prem

        if not self.on_prem:
            img_bytesio = get_bytesio_from_s3(self.s3_client, self.bucket_name, img_key)
            img_bytesio.seek(0)
            img_pil = Image.open(io.BytesIO(img_bytesio.read()))
            img_array = np.array(img_pil)

        else:
            img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)

        if self.transform:
            img_tensor = self.transform(img_array)
        else:
            img_tensor = img_array

        # Fetch one hot encoded labels for all classes of particle_type as a Series
        particle_type_tensor = row[['particle_type_Fe', 'particle_type_X-ray']]
        # Convert Series to numpy array
        particle_type_tensor = particle_type_tensor.to_numpy().astype(np.bool_)
        
        # Convert One Hot Encoded labels to tensor
        particle_type_tensor = torch.from_numpy(particle_type_tensor)
        # Convert tensor data type to Float
        particle_type_tensor = particle_type_tensor.type(torch.FloatTensor)

        return img_tensor, particle_type_tensor

def main():
    """main function to test PyTorch Dataset class"""
    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_meta_csv_path = f"{s3_path}/meta.csv"
    s3_meta_fname = "meta.csv"

    #### testing get file functions from s3 ####
    local_train_dir = data_dir / 'processed'

    #### testing dataset class ####
    train_csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    training_bps = BPSMouseDataset(train_csv_file,
                                   local_train_dir,
                                   transform=None,
                                   file_on_prem=True)
    print(training_bps.__len__())
    print(training_bps.__getitem__(0))

if __name__ == "__main__":
    main()