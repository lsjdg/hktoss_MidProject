from modules.data_provider import *

# prepare train and test set
X_train, X_test, y_train, y_test = prepare_data()

# save data as pkl
DATA_PATH = '../../data/modified/train_test_set/'


save_data(X_train, DATA_PATH + 'X_train.pkl')
save_data(X_test, DATA_PATH + 'X_test.pkl')
save_data(y_train, DATA_PATH + 'y_train.pkl')
save_data(y_test, DATA_PATH + 'y_test.pkl')


def load_train_test():
    DATA_PATH = '../../data/modified/train_test_set/'
    with open(DATA_PATH + 'X_train.pkl', 'rb') as file:
        X_train = pickle.load(file)
    with open(DATA_PATH + 'X_test.pkl', 'rb') as file:
        X_test = pickle.load(file)
    with open(DATA_PATH + 'y_train.pkl', 'rb') as file:
        y_train = pickle.load(file)
    with open(DATA_PATH + 'y_test.pkl', 'rb') as file:
        y_test = pickle.load(file)

    return X_train, X_test, y_train, y_test