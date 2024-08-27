from trainer import *

# Train model and return metrics for comparison
def get_metrics(model, X_train, y_train, X_test, y_test):
    # Train the model on the entire training data
    trained_model = fit_model(model, X_train, y_train)
    
    # Predict probabilities for the test set
    pred_proba = trained_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy, precision, recall, f1, logloss, roc_auc = calculate_metrics(y_test, pred_proba)
    
    # Store metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'logloss': logloss,
        'roc_auc': roc_auc
    }
    
    return metrics
