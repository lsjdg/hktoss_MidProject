from prepare_data import load_train_test
from modules.metrics import *
import warnings
import joblib

warnings.filterwarnings('ignore')
models = ['xgb', 'lgbm', 'rf', 'svm']

# load train, test set
X_train, X_test, y_train, y_test = load_train_test()

for model_name in models:
    model_filename = f'../parameters/best_models/{model_name}_best_model.pkl'
    loaded_model = joblib.load(model_filename)
    metrics_dict = get_metrics(loaded_model, X_train, y_train, X_test, y_test)
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(f'../experiments/metrics/{model_name}_metrics.csv', index=False)