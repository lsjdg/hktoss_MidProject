import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import os


# load data
def load_data():
    DATA_PATH = "data/given/PCA_3_df.csv"
    df = pd.read_csv(DATA_PATH)
    return df


# train test split
def prepare_data():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"], random_state=42, stratify=df["target"]
    )
    return X_train, X_test, y_train, y_test


# reduce data demension
def reduce_data(X, y, reduction_ratio=0.1):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=reduction_ratio, random_state=42)

    for _, small_index in sss.split(X, y):
        X_reduced = X.iloc[small_index]
        y_reduced = y.iloc[small_index]

    return X_reduced, y_reduced


# save data as pickle
def save_data(data, path):
    # Ensure the directory exists
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)
