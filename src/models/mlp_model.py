from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

import pyprojroot
from pyprojroot import here
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler

processed_data_dir = root / "data" / "processed"


def prepare_data(train_data : pd.DataFrame, test_data: pd.DataFrame):
    train_fnames = list(train_data['filename'])
    test_fnames = list(test_data['filename'])
    for i in range(len(train_fnames)):
        train_fnames[i] = processed_data_dir / train_fnames[i]
    for i in range(len(test_fnames)):
        test_fnames[i] = processed_data_dir / test_fnames[i]
    
    X_textual_train = train_data[['dose_Gy', 'hr_post_exposure']]
    y_train = train_data['partice_type']

    X_textual_test = test_data[['dose_Gy', 'hr_post_exposure']]
    y_test = test_data['partice_type']

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    imsize = (200, 200)
    
    X_im_train = []
    X_im_test = []

    for im_path in train_fnames:
        im = Image.open(im_path)
        im = im.resize(imsize)
        im_array = np.array(im)
        X_im_train.append(im_array)

    for im_path in test_fnames:
        im = Image.open(im_path)
        im = im.resize(imsize)
        im_array = np.array(im)
        X_im_test.append(im_array)
    
    return (X_im_train, X_im_test, 
            X_textual_train, X_textual_test, 
            y_train, y_test)






def main():
    train_csv_fname = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    test_csv_fname = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    df_train = pd.read_csv(processed_data_dir / train_csv_fname)
    df_test = pd.read_csv(processed_data_dir / test_csv_fname)
    processed_train, processed_test = prepare_data(df_train, df_test)



if __name__ == "__main__":
    main()