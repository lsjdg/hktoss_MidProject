import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# load data
def load_data():
    DATA_PATH = 'data/modified/iris.csv'
    df = pd.read_csv(DATA_PATH)
    return df

# train test split
def prepare_data():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('target', axis=1),
        df['target'],
        random_state=42,
        stratify=df['target']
    )
    return X_train, X_test, y_train, y_test

## save data as pickle
def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)