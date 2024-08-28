from modules.data_provider import *
from ml.modules.metrics import *
from modules.trainer import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from parameters.params import *

# prepare train and test set
X_train, X_test, y_train, y_test = prepare_data()

# define models
models = {
        'xgb': (XGBClassifier(), X_train, X_test, y_train, y_test),
        'lgbm': (LGBMClassifier(), X_train, X_test, y_train, y_test),
        'lr': (LogisticRegression(), X_train, X_test, y_train, y_test),
        'svm': (SVC(probability=True), X_train, X_test, y_train, y_test),
        'rf': (RandomForestClassifier(), X_train, X_test, y_train, y_test)
    }

# find best parameters by GridSearch
for model_name, (model, X_trn, X_tst, y_trn, y_tst) in models.items():
        print(f'### {model_name} ###')
        find_best_params(model, globals().get(f'{model_name}_params', {}), X_trn, y_trn)
        print('best model and parameters saved at parameters/')
        

