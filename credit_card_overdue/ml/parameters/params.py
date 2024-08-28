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

rf_params = {
    'n_estimators': 1000,  
    'max_depth': None,  
    'min_samples_split': 2,  
    'min_samples_leaf': 1,  
    'max_features': 'auto',  
    'bootstrap': True,  
    'criterion': 'gini', 
    'random_state': 2018, 
    'n_jobs': -1  
}

lr_params = {
    'penalty': 'l2',  
    'C': 1.0,  
    'solver': 'lbfgs',  
    'max_iter': 100,  
    'random_state': 2018,
    'n_jobs': -1 
}

svm_params = {
    'C': 1.0,  
    'kernel': 'rbf',  
    'gamma': 'scale',  
    'max_iter': -1,  
    'random_state': 2018
}
