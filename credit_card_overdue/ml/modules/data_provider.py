import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import os
import mlflow


# load data
def load_data(DATA_PATH="data/modified/PCA_3_df.csv"):
    df = pd.read_csv(DATA_PATH)
    return df


# train test split
def prepare_data():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"], random_state=42, stratify=df["target"]
    )
    return X_train, X_test, y_train, y_test


# reduce data demension
def reduce_data(X, y, reduction_ratio=0.1):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=reduction_ratio, random_state=42)

    for _, small_index in sss.split(X, y):
        X_reduced = X.iloc[small_index]
        y_reduced = y.iloc[small_index]

    return X_reduced, y_reduced


# save result as pkl to artifacts
def save_artifacts(file, artifact_name, clean_up):
    os.makedirs("temp_data/", exist_ok=True)
    artifact_path = f"temp_data/{artifact_name}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(file, f)
    mlflow.log_artifact(artifact_path)
    if clean_up:
        os.remove(artifact_path)


# load pkl from artifacts and deserialize
def load_artifacts(run_id, artifact_name):
    # Download the artifact file from the MLflow server
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"{artifact_name}.pkl"
    )

    # Load the artifact file's contents
    with open(artifact_path, "rb") as f:
        artifact_bytes = f.read()

    # Deserialize the artifact
    artifact_deserialized = pickle.loads(artifact_bytes)

    return artifact_deserialized
