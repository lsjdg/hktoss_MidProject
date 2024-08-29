import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.base import clone
import joblib


# find best params w/ GridSearch

def find_best_params(model, params, X_train, y_train):
    # Create GridSearchCV object with verbose parameter to show progress
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, verbose=3)
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Save best parameters to CSV
    best_params_df = pd.DataFrame([grid_search.best_params_])
    best_params_csv_path = f'ml/experiments/best_params/{type(model).__name__}_best_params.csv'
    best_params_df.to_csv(best_params_csv_path, index=False)
    
    # Save the best model to a file
    model_filename = f'ml/experiments/tuned_models/{type(model).__name__}_tuned_model.pkl'
    joblib.dump(grid_search.best_estimator_, model_filename)

    print(f'Best parameters saved to {best_params_csv_path}')
    print(f'Best model saved to {model_filename}')

    return grid_search.best_params_

# Fit the model with early stopping if applicable
def train_model(model, X_train, y_train):
    verbose = 500 if hasattr(model, 'verbose') else 0
    
    model.fit(
        X_train, y_train,
        verbose=verbose,
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


