from xgboost.sklearn import XGBClassifier, XGBRegressor 
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from CART import CART

from time import time
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

class Model:
    def __init__(self, model, task_type, params, seed=114514) -> None:
        self.task_type = task_type
        if model == 'xgb':
            if task_type[0] == 'c':
                self.model = XGBClassifier(
                    **params,
                    objective="binary:logistic",  
                    eval_metric="logloss",       
                    seed=seed
                )
            elif task_type[0] == 'r':
                self.model = XGBRegressor(
                    **params,
                    booster='gbtree',
                    objective='reg:squarederror',
                    seed=seed
                )
        elif model == 'lightgbm':
            if task_type[0] == 'c':
                self.model = LGBMClassifier(
                    **params,
                    objective='binary',
                    verbose='-1',
                    seed=seed
            )
            elif task_type[0] == 'r':
                self.model = LGBMRegressor(
                    **params,
                    verbose='-1',
                    seed=seed
            )

        elif model == 'mlp':
            if 'hidden_layer_dim' in params:
                params['hidden_layer_sizes'] = tuple((params['hidden_layer_dim'] for _ in range(params['hidden_layer_num'])))
                del params['hidden_layer_dim'] 
                del params['hidden_layer_num']
            if task_type[0] == 'c':
                self.model = MLPClassifier(
                    **params,
                    random_state=seed
                )
            elif task_type[0] == 'r':
                self.model = MLPRegressor(
                    **params,
                    random_state=seed
                )   
        elif model == 'svm':
            if task_type[0] == 'c':
                self.model = SVC(**params, random_state=seed)
            elif task_type[0] == 'r':
                self.model = SVR(**params)
            
        elif model == 'cart':
            if task_type[0] == 'c':
                self.model = CART(
                    **params,
                    cart_type='classification'
                )
            elif task_type[0] == 'r':
                self.model = CART(
                    **params,
                    cart_type='regression',
                )

        self.training_time = None
        self.inference_time = None

    def fit(self, X, y):
        sta = time()
        self.model.fit(X, y)
        self.training_time = time() - sta
    
    def predict(self, X):
        sta = time()
        pred = self.model.predict(X)
        self.inference_time = time() - sta
        return pred

    def test(self, X, y):
        pred = self.predict(X)
        if self.task_type[0] == 'c':
            metric = f1_score(y, pred)
        else:
            metric = np.sqrt(mean_squared_error(y, pred))
        return metric, self.training_time, self.inference_time
    def test_all(self, X, y):
        pred = self.predict(X)
        if self.task_type[0] == 'c':
            acc = accuracy_score(y, pred)
            pre = precision_score(y, pred)
            rec = recall_score(y, pred)
            f1 = f1_score(y, pred)
            metrics = [acc, pre, rec, f1]
        else:
            mse = np.sqrt(mean_squared_error(y, pred))
            metrics = [mse]
        return metrics

def load_data(dataID):
    traindf = pd.read_csv(f'data/{dataID}/train.csv', header=None)
    testdf = pd.read_csv(f'data/{dataID}/test.csv', header=None)
    X_train = traindf.iloc[:,:-1].values
    X_test = testdf.iloc[:,:-1].values
    if dataID == 3:
        y_train = traindf.iloc[:,-1].apply(lambda x: 1 if x=='relapse' else 0).values
        y_test = testdf.iloc[:,-1].apply(lambda x: 1 if x=='relapse' else 0).values
    elif dataID == 1:
        y_train = traindf.iloc[:,-1].apply(lambda x: 1 if x=='yes' else 0).values
        y_test = testdf.iloc[:,-1].apply(lambda x: 1 if x=='yes' else 0).values
    else:
        y_train = traindf.iloc[:,-1].values
        y_test = testdf.iloc[:,-1].values
    return (X_train), y_train, X_test, y_test





        