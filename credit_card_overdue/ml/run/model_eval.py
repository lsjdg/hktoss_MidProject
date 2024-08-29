import mlflow
import mlflow.sklearn
import pandas as pd
from ml.run.prepare_data import load_train_test
from ml.modules.metrics import *
import joblib

X_train, X_test, y_train, y_test = load_train_test()
models = ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "SVC"]

mlflow.set_experiment("credit_card_overdue_pred")
for model_name in models:
    model_filename = f"ml/experiments/best_models/{model_name}_tuned_model.pkl"

    with mlflow.start_run(run_name=f"Model_Evaluation_{model_name}") as run:
        loaded_model = joblib.load(model_filename)

        metrics_dict = get_metrics(loaded_model, X_train, y_train, X_test, y_test)

        # for key, value in metrics_dict.items():
        # mlflow.log_metric(key, value)

        # mlflow.sklearn.log_model(loaded_model, 'model')

        metrics_df = pd.DataFrame([metrics_dict])
        metrics_csv_path = f"ml/experiments/metrics/{model_name}_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        # mlflow.log_artifact(metrics_csv_path)
