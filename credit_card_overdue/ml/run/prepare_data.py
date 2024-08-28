import mlflow
import mlflow.pytorch  # 또는 mlflow.sklearn, mlflow.xgboost 등 모델에 맞는 라이브러리
import pickle
from modules.data_provider import *

# MLflow experiment 설정
mlflow.set_experiment("data_preparation_experiment")

# Prepare train and test set
with mlflow.start_run(run_name="Data Preparation") as run:
    X_train, X_test, y_train, y_test = prepare_data()

    # Save data as pkl
    DATA_PATH = '../../data/modified/train_test_set/'

    save_data(X_train, DATA_PATH + 'X_train.pkl')
    save_data(X_test, DATA_PATH + 'X_test.pkl')
    save_data(y_train, DATA_PATH + 'y_train.pkl')
    save_data(y_test, DATA_PATH + 'y_test.pkl')

    # Log data as artifacts
    mlflow.log_artifact(DATA_PATH + 'X_train.pkl')
    mlflow.log_artifact(DATA_PATH + 'X_test.pkl')
    mlflow.log_artifact(DATA_PATH + 'y_train.pkl')
    mlflow.log_artifact(DATA_PATH + 'y_test.pkl')
