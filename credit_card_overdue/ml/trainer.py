import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')

# find best params w/ GridSearch
def find_best_params(model, params, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    best_params_df = pd.DataFrame([grid_search.best_params_])
    best_params_df.to_csv(f'parameters/best_params/{type(model).__name__}.csv', index=False)
    return grid_search.best_estimator_

# Generate train and validation indices for each fold
def get_fold_indices(X, y, cv=5):
    return list(StratifiedKFold(n_splits=cv).split(X, y))

# Fit the model with early stopping if applicable
def fit_model(model, X_train, y_train, X_vld, y_vld):
    # define eval_set for XGB, LGBM
    eval_set = [(X_train, y_train), (X_vld, y_vld)] if hasattr(model, 'eval_set') else None
    verbose = 500 if hasattr(model, 'verbose') else 0
    early_stopping_rounds = 30 if hasattr(model, 'early_stopping_rounds') else None
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=verbose,
        early_stopping_rounds=early_stopping_rounds
    )
    return model

# Calculate metrics for the validation set
def calculate_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)  # set threshold as 0.5
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])

    return accuracy, precision, recall, f1, logloss, roc_auc


