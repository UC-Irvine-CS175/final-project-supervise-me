import boto3
from botocore import UNSIGNED
from botocore.config import Config

from src.data_utils import *
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))


def main():
    data_dir = f'{root}/data/raw'
    bucket_name = 'nasa-bps-training-data'
    s3_path = 'Microscopy/train'
    s3_meta_csv_path = f'{s3_path}/meta.csv'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    meta_csv_path_fname = get_file_from_s3(
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_file_path=s3_meta_csv_path,
        local_file_path=data_dir
    )

    print(meta_csv_path_fname)

    meta_train_path_fname, meta_test_path_fname = train_test_split_subset_meta_dose_hr(
        subset_meta_dose_hr_csv_path=meta_csv_path_fname,
        test_size=0.2,
        out_dir_csv=data_dir,
        stratify_col="particle_type"
    )

    """
    Saving tiffs from meta.csv for training entire dataset
    Uncomment when ready...
    """

    ## Saves tiffs from meta_train.csv
    print("Downloading tifs from meta_train.csv...")
    save_tiffs_local_from_s3(
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_path=s3_path,
        local_fnames_meta_path=meta_train_path_fname,
        save_file_path=data_dir
    )
    
    ## Saves tiffs from meta_test.csv
    print("Downloading tifs from meta_test.csv...")
    save_tiffs_local_from_s3(
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_path=s3_path,
        local_fnames_meta_path=meta_test_path_fname,
        save_file_path=data_dir
    )

    


if __name__ == '__main__':
    main()