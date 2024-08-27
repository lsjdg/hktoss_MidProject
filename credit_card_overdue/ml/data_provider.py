import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle

# load data
def load_data():
    DATA_PATH = '../data/modified/'
    df = pd.read_csv(DATA_PATH)
    return df

## ordinal encoder for ordinal data
def ordinal(labels):
    encoder = OrdinalEncoder()
    encoded_labels = encoder.fit_transform(labels.reshape(-1, 1))
    return encoded_labels

## one hot encoder for nominal data
def one_hot(labels):
    encoder = OneHotEncoder(sparse=False)
    encoded_labels = encoder.fit_transform(labels.reshape(-1, 1))
    return encoded_labels

## Standardization (for Logisitc Regression & SVM)
def standardization(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

## save data as pickle
def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)