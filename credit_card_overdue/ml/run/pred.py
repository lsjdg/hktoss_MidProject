from ml.run.prepare_data import load_train_test
from xgboost import XGBClassifier
import pandas as pd
import mlflow
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

mlflow.set_experiment("prediction")

X_train, X_test, y_train, y_test = load_train_test()
