from ml.modules.data_provider import *
import pickle
import mlflow


def load_train_test():
    DATA_PATH = "data/modified/train_test_set/"

    with open(DATA_PATH + "X_train.pkl", "rb") as file:
        X_train = pickle.load(file)

    with open(DATA_PATH + "X_test.pkl", "rb") as file:
        X_test = pickle.load(file)

    with open(DATA_PATH + "y_train.pkl", "rb") as file:
        y_train = pickle.load(file)

    with open(DATA_PATH + "y_test.pkl", "rb") as file:
        y_test = pickle.load(file)

    return X_train, X_test, y_train, y_test


mlflow.set_experiment("prepare data")

with mlflow.start_run(run_name="prepare data") as run:

    # Prepare train and test set
    X_train, X_test, y_train, y_test = prepare_data()
    X_train_reduced, y_train_reduced = reduce_data(X_train, y_train)
    X_test_reduced, y_test_reduced = reduce_data(X_test, y_test)

    # Save data as pkl
    DATA_PATH = "data/modified/train_test_set/"

    save_data(X_train_reduced, DATA_PATH + "X_train.pkl")
    save_data(X_test_reduced, DATA_PATH + "X_test.pkl")
    save_data(y_train_reduced, DATA_PATH + "y_train.pkl")
    save_data(y_test_reduced, DATA_PATH + "y_test.pkl")

    # log train, test sets
    mlflow.log_artifact(DATA_PATH + "X_train.pkl")
    mlflow.log_artifact(DATA_PATH + "X_test.pkl")
    mlflow.log_artifact(DATA_PATH + "y_train.pkl")
    mlflow.log_artifact(DATA_PATH + "y_test.pkl")
