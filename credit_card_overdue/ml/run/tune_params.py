import mlflow
from ml.modules.data_provider import load_artifacts, save_artifacts
from ml.modules.trainer import find_best_params_and_estimator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ml.modules.params import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

with open("ml/experiments/run_ids/prepare_data_id.txt", "r") as f:
    run_id = f.read().strip()


mlflow.set_experiment("param tuning")

X_train = load_artifacts(run_id, "X_train_reduced")
X_test = load_artifacts(run_id, "X_test_reduced")
y_train = load_artifacts(run_id, "y_train_reduced")
y_test = load_artifacts(run_id, "y_test_reduced")

# Define models
models = {
    "XGBClassifier": (XGBClassifier(), X_train, X_test, y_train, y_test),
    "LGBMClassifier": (LGBMClassifier(), X_train, X_test, y_train, y_test),
    "RandomForestClassifier": (
        RandomForestClassifier(),
        X_train,
        X_test,
        y_train,
        y_test,
    ),
}

# Find best parameters by GridSearch
for model_name, (model, X_trn, X_tst, y_trn, y_tst) in models.items():
    with mlflow.start_run(run_name=f"GridSearch_{model_name}") as run:
        print(f"### {model_name} ###")
        best_params, best_estimator = find_best_params_and_estimator(
            model, globals().get(f"{model_name}_params", {}), X_trn, y_trn
        )

        # Log best parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Serialize the best estimator and log it as an artifact
        save_artifacts(best_estimator, f"{model_name}_tuned", clean_up=True)

        print(f"Best model and parameters for {model_name} successfully logged")

        # Save the run_id
        run_id = run.info.run_id
        with open(f"ml/experiments/run_ids/tune_{model_name}_id.txt", "w") as f:
            f.write(run_id)
