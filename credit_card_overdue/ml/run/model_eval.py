import mlflow
import mlflow.sklearn
import pandas as pd
from ml.run.prepare_data import load_train_test
from ml.modules.trainer import get_metrics
import joblib

X_train, X_test, y_train, y_test = load_train_test()
models = ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"]
mlflow.set_experiment("model evaluation")

for model_name in models:
    model_filename = f"ml/experiments/tuned_models/{model_name}_tuned_model.pkl"
    loaded_model = joblib.load(model_filename)
    
    with mlflow.start_run(run_name=f"Model_Evaluation_{model_name}"):
        # Predict probabilities for test data
        y_pred_proba = loaded_model.predict_proba(X_test)

        # Get metrics
        metrics_dict = get_metrics(y_test, y_pred_proba)

        # Log metrics to MLflow
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)

        # Log the model to MLflow
        mlflow.sklearn.log_model(loaded_model, 'model')

        # Save metrics as CSV and log as an artifact
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_csv_path = f"ml/experiments/metrics/{model_name}_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        mlflow.log_artifact(metrics_csv_path)