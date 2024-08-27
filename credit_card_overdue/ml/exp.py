from trainer import *

# Train the model with cross-validation and return metrics for comparison
def get_metrics(model, X_train, y_train, cv=5):
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'logloss': [], 'roc_auc': []}
    
    fold_indices = get_fold_indices(X_train, y_train, cv)
    
    for n, (train_idx, vld_idx) in enumerate(fold_indices):
        print(f"fold {n+1}/{cv}..........")
        
        X_trn, X_vld = X_train.iloc[train_idx], X_train.iloc[vld_idx]
        y_trn, y_vld = y_train.iloc[train_idx], y_train.iloc[vld_idx]
        
        model_copy = clone(model)
        model_copy = fit_model(model_copy, X_trn, y_trn, X_vld, y_vld)
        
        y_vld_pred_proba = model_copy.predict_proba(X_vld)
        
        accuracy, precision, recall, f1, logloss, roc_auc = calculate_metrics(y_vld, y_vld_pred_proba)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['logloss'].append(logloss)
        metrics['roc_auc'].append(roc_auc)
        
    for metric, values in metrics.items():
        print(f'5 fold {metric} mean: ', np.mean(values))
    
    return metrics
