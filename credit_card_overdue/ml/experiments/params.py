LGBMClassifier_params = {
    'objective': ['multiclass'],
    'boosting_type': ['gbdt'],
    'eval_metric': ['logloss'],
    'n_estimators': [1000, 5000],
    'max_depth': [10, 20],
    'num_leaves': [31, 63],
    'learning_rate': [0.05, 0.1],
    'min_data_in_leaf': [20, 40],
    'feature_fraction': [0.8, 0.9],
    'bagging_fraction': [0.8, 0.9],
    'seed': [2018],
    'verbose': [-1],
    'n_jobs': [-1],
}

XGBClassifier_params = {
    'booster': ['gbtree'],
    'tree_method': ['hist'],
    'objective': ['multi:softprob'],
    'eval_metric': ['mlogloss'],
    'n_estimators': [3000, 1000],
    'max_depth': [7, 12],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.8],
    'lambda': [0.01, 0.1],
    'alpha': [0.5, 2],
    'seed': [2018]
}


RandomForestClassifier_params = {
    'n_estimators': [500, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True],
    'criterion': ['gini'],
    'random_state': [2018]
}

SVC_params = {
    'C': [1.0, 10.0],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale'],
    'max_iter': [1000],
    'random_state': [2018]
}
