from prepare_data import load_train_test
from ml.modules.metrics import *
from modules.trainer import *
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from experiments.params import *

# load train, test set
X_train, X_test, y_train, y_test = load_train_test()

# define models
models = {
        'xgb': (XGBClassifier(), X_train, X_test, y_train, y_test),
        'lgbm': (LGBMClassifier(), X_train, X_test, y_train, y_test),
        'svm': (SVC(probability=True), X_train, X_test, y_train, y_test),
        'rf': (RandomForestClassifier(), X_train, X_test, y_train, y_test)
    }

# find best parameters by GridSearch
for model_name, (model, X_trn, X_tst, y_trn, y_tst) in models.items():
        print(f'### {model_name} ###')
        find_best_params(model, globals().get(f'{model_name}_params', {}), X_trn, y_trn)
        print('best model and parameters saved at parameters/')
        

