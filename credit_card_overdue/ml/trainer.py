import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

def train_model(model, train, test, cv = 5):
    test_preds = []
    vld_preds = []
    feat_importance = np.zeros(test.shape[1])    
    for n, (train_idx, vld_idx) in enumerate(StratifiedKFold(cv).split(train.drop('credit', axis = 1).values, train['credit'].values)):
        print(f"fold {n+1}/{cv}..........")        
        X_train = train.loc[train_idx, :].drop('credit', axis = 1)
        X_vld = train.loc[vld_idx, :].drop('credit', axis = 1)
        y_train = train.loc[train_idx, 'credit'].values
        y_vld = train.loc[vld_idx, 'credit'].values
        
        model.fit(
            X_train, y_train,
            eval_set = [(X_train, y_train), (X_vld, y_vld)],
            verbose = 500, early_stopping_rounds = 30
        )    
        vld_preds.append(log_loss(y_vld, model.predict_proba(X_vld)))        
        
        test_pred = model.predict_proba(test)
        test_preds.append(test_pred)
        feat_importance += model.feature_importances_                        
        
    feat_importance = feat_importance / cv
    feat_importance = pd.Series(data = feat_importance, index = test.columns)

    print('5 fold logloss mean: ', np.mean(vld_preds))
    return test_preds, feat_importance