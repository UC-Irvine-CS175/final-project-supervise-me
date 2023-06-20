from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, jaccard_score, hamming_loss

import pyprojroot
from pyprojroot import here
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

processed_data_dir = root / "data" / "processed"


def prepare_data(train_csv_fname : str, test_csv_fname: str, multi_label: bool):
    """
    Prepares data from train/test csv files for training
    
    Args:
        train_csv_fname (str): csv file name for train data
        test_csv_fname (str): csv file name for test data
        multi_label (bool, optional): Whether to return multi-label data. Defaults to False.
    """
    
    # Reads in csv data to a pandas dataframe
    train_data = pd.read_csv(processed_data_dir / train_csv_fname)
    test_data = pd.read_csv(processed_data_dir / test_csv_fname)

    # Takes filename column and converts it into a path where files are located
    train_fnames = list(train_data['filename'])
    test_fnames = list(test_data['filename'])
    for i in range(len(train_fnames)):
        train_fnames[i] = processed_data_dir / train_fnames[i]
    for i in range(len(test_fnames)):
        test_fnames[i] = processed_data_dir / test_fnames[i]
    
    # Separates the rest of the categorical features and labels

    if not multi_label:
        label_encoder = LabelEncoder()
        y_train = train_data['particle_type']
        y_test = test_data['particle_type']
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)
    else:
        oh_encoder = OneHotEncoder()
        y_train = oh_encoder.fit_transform(train_data[['dose_Gy', 'particle_type']])
        y_test = oh_encoder.fit_transform(test_data[['dose_Gy','particle_type']])
    
    # Converts images into numpy array
    imsize = (200, 200)
    len_train = len(train_fnames)
    len_test = len(test_fnames)
    X_im_train = np.empty((len_train,) + imsize + (1,), dtype=np.uint8)
    X_im_test = np.empty((len_test,) + imsize + (1,), dtype=np.uint8)
    count = 0
    for i, im_path in enumerate(train_fnames):
        count += 1
        im = Image.open(im_path)
        im = im.convert('L')
        im = im.resize(imsize)
        im_array = np.array(im)
        X_im_train[i, :, :, 0] = im_array
        if count % 100 == 0:
            print(count)

    for i, im_path in enumerate(test_fnames):
        count += 1
        im = Image.open(im_path)
        im = im.convert('L')
        im = im.resize(imsize)
        im_array = np.array(im)
        X_im_test[i, :, :, 0] = im_array
        if count % 100 == 0:
            print(count)

    # Flattens image data
    X_train = X_im_train.reshape(X_im_train.shape[0], -1)
    X_test = X_im_test.reshape(X_im_test.shape[0], -1)
   
    # Scales features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encodes labels
    

    return (X_train, X_test, y_train, y_test)

def train_and_evaluate_model(X_tr, X_te, y_tr, y_te, multi_label):
    # Trains model incrementally over 10 epochs to 
    # allow checking for losses and accuracies over time.   

    if multi_label:
        # MLPCLassifier's partial_fit() does not support multi-label classification
        model = MLPClassifier(hidden_layer_sizes=(25,25), batch_size=32,
                        max_iter=15, learning_rate_init=3e-4, random_state=42)
        model.fit(X_tr, y_tr)
        
        tr_pred = model.predict(X_tr)
        tr_acc = accuracy_score(y_tr, tr_pred)
        train_loss = log_loss(y_tr, model.predict_proba(X_tr))

        val_pred = model.predict(X_te)
        val_acc = accuracy_score(y_te, val_pred)
        val_loss = log_loss(y_te, model.predict_proba(X_te))

        print('jaccard score:', jaccard_score(y_te, val_pred, average='micro'))
        print('hamming loss:', hamming_loss(y_te, val_pred))

        return tr_acc, val_acc, train_loss, val_loss
    else:
        model = MLPClassifier(hidden_layer_sizes=(25,25), batch_size=64,
                        max_iter=1, learning_rate_init=3e-4, random_state=42)
        epochs = 10
        train_accs = []
        val_accs = []
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            print(f"Epoch {epoch}")

            model.partial_fit(X_tr, y_tr, classes=np.unique(y_tr))
            
            tr_pred = model.predict(X_tr)
            tr_acc = accuracy_score(y_tr, tr_pred)
            train_accs.append(tr_acc)
            
            train_loss = log_loss(y_tr, model.predict_proba(X_tr))
            train_losses.append(train_loss)

            val_pred = model.predict(X_te)
            val_acc = accuracy_score(y_te, val_pred)
            val_accs.append(val_acc)
            
            val_loss = log_loss(y_te, model.predict_proba(X_te))
            val_losses.append(val_loss)
    
        return train_accs, val_accs, train_losses, val_losses

def plot_results(train_accs, val_accs, train_losses, val_losses):
    epochs = range(1, 11)
    
    # Plots tr/val accuracies over epochs
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.show()

    # Plots tr/val losses over epochs
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()

def main():
    eval_names = ['train_acc', 'val_acc', 'train_loss', 'val_loss']
    train_csv_fname = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    test_csv_fname = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    
    multi_label = input('Multi-label? (y/n): ')
    X_tr, X_te, y_tr, y_te = prepare_data(train_csv_fname, test_csv_fname, multi_label)
    tr_accs, val_accs, tr_losses, val_losses = train_and_evaluate_model(X_tr, X_te, y_tr, y_te, multi_label)
    
    if multi_label == 'y':
        classification_type = 'Multi-label'
    else:
        classification_type = 'Single-label'
        plot_results(tr_accs, val_accs, tr_losses, val_losses)

    print(f"{classification_type} statistics:")
    for i, val in enumerate([tr_accs, val_accs, tr_losses, val_losses]):
        if multi_label == 'y':
            print(f"{eval_names[i]}: {val}")
        else:
            print(f"{eval_names[i]}: {val[-1]}")

if __name__ == "__main__":
    main()