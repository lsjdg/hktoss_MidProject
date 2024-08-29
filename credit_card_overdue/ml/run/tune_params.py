import mlflow
from ml.run.prepare_data import load_train_test
from ml.modules.metrics import *
from ml.modules.trainer import find_best_params
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from ml.experiments.params import *
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

mlflow.set_experiment('parameter_tuning')

X_train, X_test, y_train, y_test = load_train_test()

# Define models
models = {
    'XGBClassifier': (XGBClassifier(), X_train, X_test, y_train, y_test),
    'LGBMClassifier': (LGBMClassifier(), X_train, X_test, y_train, y_test),
    'SVC': (SVC(probability=True), X_train, X_test, y_train, y_test),
    'RandomForestClassifier': (RandomForestClassifier(), X_train, X_test, y_train, y_test)
}

# Find best parameters by GridSearch
for model_name, (model, X_trn, X_tst, y_trn, y_tst) in models.items():
    with mlflow.start_run(run_name=f'GridSearch_{model_name}') as run:
        print(f'### {model_name} ###')
        best_params = find_best_params(model, globals().get(f'{model_name}_params', {}), X_trn, y_trn)
        
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Log best params CSV file and best model files
        best_params_csv_path = f'ml/experiments/best_params/{model_name}_best_params.csv'
        model_filename = f'ml/experiments/tuned_models/{model_name}_tuned_model.pkl'

        if os.path.isfile(best_params_csv_path):
            mlflow.log_artifact(best_params_csv_path)
            print('best params logged at artifacts')

        if os.path.isfile(model_filename):
            mlflow.log_artifact(model_filename)
            print('tuned model logged at artifacts')

        print(f'Best model and parameters for {model_name} saved at parameters/')
