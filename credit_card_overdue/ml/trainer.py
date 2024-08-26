import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings('ignore')

def get_fold_indices(X, y, cv=5):
    """Generate train and validation indices for each fold."""
    return list(StratifiedKFold(n_splits=cv).split(X, y))

def fit_model(model, X_train, y_train, X_vld, y_vld):
    """Fit the model with early stopping if applicable."""
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

def calculate_log_loss(y_vld, y_vld_pred_proba):
    """Calculate log loss for the validation set."""
    return log_loss(y_vld, y_vld_pred_proba)

def calculate_feature_importance(model, test_columns, cv):
    """Calculate the average feature importance across folds."""
    if hasattr(model, 'feature_importances_'):
        feat_importance = np.zeros(len(test_columns))
        feat_importance += model.feature_importances_
        feat_importance /= cv
        return pd.Series(data=feat_importance, index=test_columns)
    else:
        return None

def train_model(model, train, test, cv=5):
    """Train the model with cross-validation and return predictions and feature importance."""
    test_preds = []
    vld_preds = []
    
    X = train.drop('credit', axis=1)
    y = train['credit']
    fold_indices = get_fold_indices(X, y, cv)
    
    for n, (train_idx, vld_idx) in enumerate(fold_indices):
        print(f"fold {n+1}/{cv}..........")
        
        X_train, X_vld = X.iloc[train_idx], X.iloc[vld_idx]
        y_train, y_vld = y.iloc[train_idx], y.iloc[vld_idx]
        
        model = fit_model(model, X_train, y_train, X_vld, y_vld)
        
        y_vld_pred_proba = model.predict_proba(X_vld)
        vld_preds.append(calculate_log_loss(y_vld, y_vld_pred_proba))
        
        test_pred = model.predict_proba(test)
        test_preds.append(test_pred)
    
    feat_importance = calculate_feature_importance(model, X.columns, cv)
    
    print('5 fold logloss mean: ', np.mean(vld_preds))
    return test_preds, feat_importance
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings('ignore')

def get_fold_indices(X, y, cv=5):
    """Generate train and validation indices for each fold."""
    return list(StratifiedKFold(n_splits=cv).split(X, y))

def fit_model(model, X_train, y_train, X_vld, y_vld):
    """Fit the model with early stopping if applicable."""
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

def calculate_log_loss(y_vld, y_vld_pred_proba):
    """Calculate log loss for the validation set."""
    return log_loss(y_vld, y_vld_pred_proba)

def calculate_feature_importance(model, test_columns, cv):
    """Calculate the average feature importance across folds."""
    if hasattr(model, 'feature_importances_'):
        feat_importance = np.zeros(len(test_columns))
        feat_importance += model.feature_importances_
        feat_importance /= cv
        return pd.Series(data=feat_importance, index=test_columns)
    else:
        return None

def train_model(model, train, test, cv=5):
    """Train the model with cross-validation and return predictions and feature importance."""
    test_preds = []
    vld_preds = []
    
    X = train.drop('credit', axis=1)
    y = train['credit']
    fold_indices = get_fold_indices(X, y, cv)
    
    for n, (train_idx, vld_idx) in enumerate(fold_indices):
        print(f"fold {n+1}/{cv}..........")
        
        X_train, X_vld = X.iloc[train_idx], X.iloc[vld_idx]
        y_train, y_vld = y.iloc[train_idx], y.iloc[vld_idx]
        
        model = fit_model(model, X_train, y_train, X_vld, y_vld)
        
        y_vld_pred_proba = model.predict_proba(X_vld)
        vld_preds.append(calculate_log_loss(y_vld, y_vld_pred_proba))
        
        test_pred = model.predict_proba(test)
        test_preds.append(test_pred)
    
    feat_importance = calculate_feature_importance(model, X.columns, cv)
    
    print('5 fold logloss mean: ', np.mean(vld_preds))
    return test_preds, feat_importance
