from ml.modules.data_provider import *
import mlflow

mlflow.set_experiment("prepare data")

with mlflow.start_run(run_name="prepare data") as run:

    # Prepare train and test set
    X_train, X_test, y_train, y_test = prepare_data()
    X_train_reduced, y_train_reduced = reduce_data(X_train, y_train)
    X_test_reduced, y_test_reduced = reduce_data(X_test, y_test)

    # Save artifacts
    save_artifacts(X_train_reduced, "X_train_reduced", clean_up=True)
    save_artifacts(X_test_reduced, "X_test_reduced", clean_up=True)
    save_artifacts(y_train_reduced, "y_train_reduced", clean_up=True)
    save_artifacts(y_test_reduced, "y_test_reduced", clean_up=True)

    # Save the run_id
    run_id = run.info.run_id
    with open("ml/experiments/run_ids/prepare_data_id.txt", "w") as f:
        f.write(run_id)
