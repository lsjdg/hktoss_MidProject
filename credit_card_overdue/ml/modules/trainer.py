from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


# find best params w/ GridSearch


def find_best_params_and_estimator(model, params, X_train, y_train):
    # Create GridSearchCV object with verbose parameter to show progress
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, verbose=3)

    # Fit the model
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_, grid_search.best_estimator_


# Fit the model with early stopping if applicable
def train_model(model, X_train, y_train):
    verbose = 500 if hasattr(model, "verbose") else 0

    model.fit(
        X_train,
        y_train,
        verbose=verbose,
    )
    return model


# Calculate metrics


def get_metrics(y_test, y_pred_proba):
    y_pred = y_pred_proba.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "log_loss": logloss,
        "roc_auc": roc_auc,
    }
