import mlflow
import pandas as pd
from ml.modules.data_provider import *
from ml.modules.trainer import get_metrics
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

mlflow.set_experiment("prediction_total")

df = pd.read_csv("data/modified/PCA_3_df.csv")
X_test = df.drop("target", axis=1)
y_test = df["target"]

with open(f"ml/experiments/run_ids/tune_XGBClassifier_id.txt", "r") as f:
    xgb_run_id = f.read().strip()
xgb_clf = load_artifacts(xgb_run_id, "XGBClassifier_tuned")

with mlflow.start_run(run_name="pred_xgb") as run:
    y_pred_proba = xgb_clf.predict_proba(X_test)
    y_pred = y_pred_proba.argmax(axis=1)

    metrics_dict = get_metrics(y_test, y_pred_proba)
    # Log metrics to MLflow
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value)

    pred_df = pd.DataFrame(y_pred, columns=["pred"])
    save_artifacts(pred_df, "pred_df", clean_up=True)
