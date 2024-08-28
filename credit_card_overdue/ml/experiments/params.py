lgbm_params = {
    'objective': ['multiclass'],
    'boosting_type': ['gbdt'],
    'eval_metric': ['logloss'],    
    'n_estimators': [10000, 5000, 2000],
    'early_stopping_round': [100], 
    'max_depth': [-1, 10, 20],
    'max_bin': [255, 128],
    'boost_from_average' : [False, True],
    'bagging_freq' : [1],
    'min_data_in_leaf': [40, 20, 10],    
    'learning_rate': [0.02272, 0.01, 0.1],    
    'num_leaves': [64, 128, 32],    
    'feature_fraction': [0.89387, 0.8, 0.9],
    'bagging_fraction': [0.76326, 0.7, 0.8],        
    'seed': [2018],
    'verbose': [-1],
    'n_jobs': [-1],    
}

xgb_params = {
    'booster' : ['gbtree'],
    'tree_method' : ['gpu_hist', 'hist'],
    'predictor' : ['gpu_predictor', 'cpu_predictor'],    
    'objective' : ['multi:softprob'],
    'eval_metric' : ['mlogloss'],
    'n_estimators' : [5000, 3000, 1000],
    'max_depth' : [9, 7, 12],
    'min_child_weight' : [5, 3, 7],    
    'learning_rate' : [0.012727, 0.01, 0.1],    
    'subsample' : [0.91020, 0.8, 0.9],
    'colsample_bytree' : [0.77959, 0.7, 0.8],    
    'colsample_bylevel' : [0.64898, 0.6, 0.7],
    'lambda' : [0.05, 0.01, 0.1],
    'alpha' : [1, 0.5, 2],    
    'seed' : [2018]
}

rf_params = {
    'n_estimators': [1000, 500, 200],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2'],  
    'bootstrap': [True, False],  
    'criterion': ['gini', 'entropy'], 
    'random_state': [2018], 
    'n_jobs': [-1]  
}

svm_params = {
    'C': [1.0, 0.1, 10.0],  
    'kernel': ['rbf', 'linear', 'poly'],  
    'gamma': ['scale', 'auto'],  
    'max_iter': [-1, 1000, 5000],  
    'random_state': [2018]
}
