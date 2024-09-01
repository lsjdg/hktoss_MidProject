LGBMClassifier_params = {
    "n_estimators": [1000],
    "max_depth": [10, 20],
    "num_leaves": [31],
    "learning_rate": [0.05, 0.1],
    "feature_fraction": [0.8],
    "bagging_fraction": [0.8],
    "min_data_in_leaf": [20],
    "seed": [2018],
    "n_jobs": [-1],
}


XGBClassifier_params = {
    "n_estimators": [1000],
    "max_depth": [7, 12],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "lambda": [0.01],
    "alpha": [0.5],
    "seed": [2018],
}


RandomForestClassifier_params = {
    "n_estimators": [500],
    "max_depth": [10, 20],
    "min_samples_split": [5],
    "min_samples_leaf": [2],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True],
    "random_state": [2018],
}
