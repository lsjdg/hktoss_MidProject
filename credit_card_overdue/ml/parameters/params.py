lgbm_params = {
    'objective': 'multiclass',
    'boosting_type': 'gbdt',
    'eval_metric' : 'logloss',    
    'n_estimators': 10000,
    'early_stopping_round': 100, 
    'max_depth': -1,
    'max_bin': 255,
    'boost_from_average' : False,
    'bagging_freq' : 1,
    'min_data_in_leaf': 40,    
    'learning_rate': 0.02272,    
    'num_leaves': 64,    
    'feature_fraction': 0.89387,
    'bagging_fraction': 0.76326,        
    'seed': 2018,
    'verbose': -1,
    'n_jobs': -1,    
}

xgb_params = {
    'booster' : 'gbtree',
    'tree_method' : 'gpu_hist',
    'predictor' : 'gpu_predictor',    
    'objective' : 'multi:softprob',
    'eval_metric' : 'mlogloss',
    'n_estimators' : 5000,
    'max_depth' : 9,
    'min_child_weight' : 5,    
    'learning_rate' : 0.012727,    
    'subsample' : 0.91020,
    'colsample_bytree' : 0.77959,    
    'colsample_bylevel' : 0.64898,
    'lambda' : 0.05,
    'alpha' : 1,    
    'seed' : 2018
}

random_forest_params = {
    'n_estimators': 1000,  # 트리의 개수
    'max_depth': None,  # 트리의 최대 깊이 (None이면 깊이 제한 없음)
    'min_samples_split': 2,  # 내부 노드를 분할하는 데 필요한 최소 샘플 수
    'min_samples_leaf': 1,  # 리프 노드에 필요한 최소 샘플 수
    'max_features': 'auto',  # 각 트리에서 사용할 최대 특성 수
    'bootstrap': True,  # 부트스트랩 샘플링 사용 여부
    'criterion': 'gini',  # 분할 품질 측정 함수 (또는 'entropy')
    'random_state': 2018,  # 난수 생성기 시드
    'n_jobs': -1  # 모든 CPU 코어 사용
}

logistic_regression_params = {
    'penalty': 'l2',  # L2 정규화 사용 (일반적으로 사용됨)
    'C': 1.0,  # 규제 강도 (낮을수록 강한 규제, 기본값은 1.0)
    'solver': 'lbfgs',  # 최적화 알고리즘 (일반적으로 사용됨)
    'max_iter': 100,  # 최대 반복 횟수
    'random_state': 2018,
    'n_jobs': -1  # 모든 CPU 코어 사용
}

svm_params = {
    'C': 1.0,  # 규제 강도 (낮을수록 강한 규제)
    'kernel': 'rbf',  # RBF 커널 사용 (일반적으로 사용됨)
    'gamma': 'scale',  # 커널 계수 (데이터의 특징에 따라 'auto' 또는 'scale' 사용)
    'max_iter': -1,  # 최대 반복 횟수 (제한 없음)
    'random_state': 2018
}
