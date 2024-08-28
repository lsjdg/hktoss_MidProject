import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.base import clone
import joblib


# find best params w/ GridSearch
def find_best_params(model, params, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    best_params_df = pd.DataFrame([grid_search.best_params_])
    best_params_df.to_csv(f'../parameters/best_params/{type(model).__name__}.csv', index=False)
    model_filename = f'../parameters/best_models/{type(model).__name__}_best_model.pkl'
    joblib.dump(grid_search.best_estimator_, model_filename)

# Fit the model with early stopping if applicable
def train_model(model, X_train, y_train, X_vld, y_vld):
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
    y_pred = y_pred_proba.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    logloss = log_loss(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

    return accuracy, precision, recall, f1, logloss, roc_auc


