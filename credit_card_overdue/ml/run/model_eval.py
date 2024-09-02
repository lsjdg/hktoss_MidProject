import mlflow
import mlflow.sklearn
from ml.modules.data_provider import load_artifacts, save_artifacts
from ml.modules.trainer import get_metrics

with open("ml/experiments/run_ids/prepare_data_id.txt", "r") as f:
    run_id = f.read().strip()

X_test = load_artifacts(run_id, "X_test_reduced")
y_test = load_artifacts(run_id, "y_test_reduced")

models = ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"]
mlflow.set_experiment("model evaluation")


for model_name in models:
    with open(f"ml/experiments/run_ids/tune_{model_name}_id.txt", "r") as f:
        run_id = f.read().strip()
    loaded_model = load_artifacts(run_id, f"{model_name}_tuned")
    print(f"{model_name} loaded from {run_id}")

    with mlflow.start_run(run_name=f"Model_Evaluation_{model_name}"):
        # Predict probabilities for test data
        y_pred_proba = loaded_model.predict_proba(X_test)
        print("pred done")
        # Get metrics
        metrics_dict = get_metrics(y_test, y_pred_proba)
        print("metrics calculated")
        # Log metrics to MLflow
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)
        print("metrics logged")
        print(f"experiment for {model_name} done")
