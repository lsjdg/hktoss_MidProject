import mlflow
import mlflow.sklearn
import pandas as pd
from prepare_data import load_train_test
from modules.metrics import *
import warnings
import joblib

warnings.filterwarnings('ignore')
models = ['xgb', 'lgbm', 'rf', 'svm']

# MLflow experiment 설정
mlflow.set_experiment("my_experiment")  # 적절한 실험 이름으로 설정

# load train, test set
X_train, X_test, y_train, y_test = load_train_test()

for model_name in models:
    model_filename = f'../parameters/best_models/{model_name}_best_model.pkl'
    loaded_model = joblib.load(model_filename)
    
    # MLflow run 시작
    with mlflow.start_run() as run:
        # 메트릭 계산
        metrics_dict = get_metrics(loaded_model, X_train, y_train, X_test, y_test)
        
        # 메트릭 기록
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)
        
        # 모델 저장
        mlflow.sklearn.log_model(loaded_model, "model")
        
        # 메트릭을 CSV로 저장
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_csv_path = f'../experiments/metrics/{model_name}_metrics.csv'
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        # MLflow에서 CSV 파일 로그
        mlflow.log_artifact(metrics_csv_path)
