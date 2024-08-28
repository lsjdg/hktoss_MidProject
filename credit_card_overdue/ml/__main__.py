from parameters.params import *
from modules.data_provider import load_data, standardization, save_data
from modules.trainer import find_best_params
from ml.modules.metrics import get_metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. 데이터 로드 및 전처리
    df = load_data()
    X = df.drop('target', axis=1)  # target 컬럼을 제외한 피처들
    y = df['target']  # target 컬럼

    # 데이터 표준화 (Logistic Regression & SVM에 적용)
    X_standardized = standardization(X)

    # 2. 훈련 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_standardized, y, test_size=0.2, random_state=42, stratify=y)

    # 3. 모델 설정
    models = {
        'XGBoost': (XGBClassifier(xgb_params), X_train, X_test, y_train, y_test),
        'LightGBM': (LGBMClassifier(lgbm_params), X_train, X_test, y_train, y_test),
        'LogisticRegression': (LogisticRegression(logistic_regression_params), X_train_std, X_test_std, y_train_std, y_test_std),
        'SVM': (SVC(svm_params, probability=True), X_train_std, X_test_std, y_train_std, y_test_std),
        'RandomForest': (RandomForestClassifier(random_forest_params), X_train, X_test, y_train, y_test)
    }

    # 4. 각 모델에 대한 그리드 서치 및 성능 평가
    for model_name, (model, X_tr, X_te, y_tr, y_te) in models.items():
        print(f'### {model_name} ###')
        best_model = find_best_params(model, getattr(params, f'{model_name.lower()}_params'), X_tr, y_tr)
        metrics = get_metrics(best_model, X_tr, y_tr, X_te, y_te)
        
        print(f'Best Params: {best_model.get_params()}')
        print(f'Metrics: {metrics}')

if __name__ == "__main__":
    main()
