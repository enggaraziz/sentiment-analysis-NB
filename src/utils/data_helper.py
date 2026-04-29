import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

def load_csv(path_to_csv):
    return pd.read_csv(path_to_csv)

def load_object(path):
    with open(path, 'rb') as reader:
        vectors = pickle.load(reader)
    return vectors

def split_train_valid(X, Y, ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=ratio,
        random_state=109
    )
    train_resources = {}
    train_resources['train_data'] = X_train
    train_resources['train_label'] = y_train
    train_resources['validation_data'] = X_test
    train_resources['validation_label'] = y_test
    return train_resources
