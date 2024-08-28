import mlflow
from prepare_data import load_train_test
from modules.metrics import *
from modules.trainer import find_best_params
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from experiments.params import *

# MLflow experiment 설정
mlflow.set_experiment('parameter_tuning')

# Load train, test set
X_train, X_test, y_train, y_test = load_train_test()

# Define models
models = {
    'xgb': (XGBClassifier(), X_train, X_test, y_train, y_test),
    'lgbm': (LGBMClassifier(), X_train, X_test, y_train, y_test),
    'svm': (SVC(probability=True), X_train, X_test, y_train, y_test),
    'rf': (RandomForestClassifier(), X_train, X_test, y_train, y_test)
}

# Find best parameters by GridSearch
for model_name, (model, X_trn, X_tst, y_trn, y_tst) in models.items():
    with mlflow.start_run(run_name=f'GridSearch_{model_name}') as run:
        print(f'### {model_name} ###')
        best_params = find_best_params(model, globals().get(f'{model_name}_params', {}), X_trn, y_trn)
        
        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Save best model
        model_filename = f'../parameters/best_models/{model_name}_best_model.pkl'
        joblib.dump(model, model_filename)
        mlflow.log_artifact(model_filename)
        
        print('Best model and parameters saved at parameters/')
