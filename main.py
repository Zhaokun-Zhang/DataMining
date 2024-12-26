from model import Model
import pandas as pd
import numpy as np
import optuna, random, json, os
from pprint import pprint
ENABLE_TRAIN=False

def setRandomSeed(seed=114514):
    random.seed(seed)
    np.random.seed(seed)

setRandomSeed()

def load_data(data_id, fold_id):
    traindf = pd.read_csv(f'data/{data_id}/train{fold_id}.csv', header=None)
    testdf = pd.read_csv(f'data/{data_id}/test{fold_id}.csv', header=None)
    X_train = traindf.iloc[:,:-1].values
    X_test = testdf.iloc[:,:-1].values
    if data_id == 3:
        y_train = traindf.iloc[:,-1].apply(lambda x: 1 if x=='relapse' else 0).values
        y_test = testdf.iloc[:,-1].apply(lambda x: 1 if x=='relapse' else 0).values
    elif data_id == 1:
        y_train = traindf.iloc[:,-1].apply(lambda x: 1 if x=='yes' else 0).values
        y_test = testdf.iloc[:,-1].apply(lambda x: 1 if x=='yes' else 0).values
    else:
        y_train = traindf.iloc[:,-1].values
        y_test = testdf.iloc[:,-1].values
    return (X_train), y_train, X_test, y_test

def test(data_id, model_name, params):
    if data_id == 2: task_type = 'regression'
    else: task_type = 'classification'

    
    if data_id != 1:
        results = []
        for fold_id in range(5):
            x1, y1, x2, y2 = load_data(data_id, fold_id)
            model = Model(model=model_name, task_type=task_type, params=params)
            model.fit(x1, y1)
            metrics = model.test(x2, y2)
            results.append(metrics)
        return np.array(results)
    else:
        x1, y1, x2, y2 = load_data(data_id, '')
        model = Model(model=model_name, task_type=task_type, params=params)
        model.fit(x1, y1)
        if ENABLE_TRAIN: print(model.test(x1, y1))
        metrics = model.test(x2, y2)
        return np.array([metrics])
    


def tune_params(trial, data_id, model_name):
    params = {}
    if model_name == 'xgb':
        params['max_depth'] = trial.suggest_int('max_depth', 8, 12) 
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 3, 5)
        params['gamma'] = trial.suggest_float('gamma', 0, 1) 
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 2, 5) 
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 5, 10) 
        params['max_delta_step'] = trial.suggest_categorical('max_delta_step', [1])
    elif model_name == 'lightgbm':
        params['num_leaves'] = trial.suggest_int('num_leaves', 5, 100) # 31
        params['max_depth'] =  trial.suggest_int('max_depth', 5, 100)# # -1 
        params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 5, 100)  #[2,50]     
        params['min_sum_hessian_in_leaf'] = trial.suggest_float('min_sum_hessian_in_leaf', 1e-5, 1e-1, log=True)  #[2,50]     
        params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1, 8)
        params['lambda_l1'] = trial.suggest_float('lambda_l1', 1e-4, 1, log=True) 
        params['lambda_l2'] = trial.suggest_float('lambda_l2', 1e-4, 1, log=True) 
        params['min_gain_to_split'] = trial.suggest_float('min_gain_to_split', 1e-4, 1, log=True) 
        params['metric'] =  trial.suggest_categorical('metric', ['f1-score'])
        params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.8, 1)
        params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.9, 1)
        
    elif model_name == 'mlp':
        params['hidden_layer_num'] = trial.suggest_categorical('hidden_layer_num', [2,4,6,8])
        params['hidden_layer_dim'] = trial.suggest_categorical('hidden_layer_dim', [32, 64, 128, 256])
        params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True)
        params['alpha'] = trial.suggest_float('alpha', 0.0001, 0.01, log=True)
        params['max_iter'] =  trial.suggest_int('max_iter', 200, 400)# # -1 
        params['early_stopping'] = trial.suggest_categorical('early_stopping', [True])
        params['validation_fraction'] = trial.suggest_float('validation_fraction', 0.1, 0.3)

    elif model_name == 'svm':
        params['C'] = trial.suggest_float('C', 0.01, 100, log=True)
        params['gamma'] = trial.suggest_float('gamma', 0.01, 10, log=True)
        params['tol'] = trial.suggest_float('tol', 1e-5, 1e-3, log=True)
        if data_id != 2: # SVC
            params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced'])

    elif model_name == 'cart':
        params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 1, 40)
        
    results = test(data_id, model_name, params).mean(axis=0)
    return results[0]

def tune_model(data_id, model_name, trial_num=100):
    if data_id == 2: study = optuna.create_study(direction="minimize") 
    else: study = optuna.create_study(direction="maximize")  
    study.optimize(lambda trial:tune_params(trial, data_id, model_name), n_trials=trial_num)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: ")
    trial.params['Value'] = trial.value
    pprint(trial.params)
    with open(f'results/{data_id}-{model_name}_tune-results', 'a') as f:
        f.write(json.dumps(trial.params)+'\n')



def load_and_select(data_id, model_name):
    if os.path.exists(f'results/{data_id}-{model_name}_tune-results'):
        with open(f'results/{data_id}-{model_name}_tune-results') as f:
            lines = f.readlines()
    else: return {}
    if len(lines) == 0: return {}

    if data_id == 2:  # find minimum
        best_val = float('inf')
        cmp = lambda a, b: a < b
    else:  # find maximum
        best_val = float('-inf')
        cmp = lambda a, b: a > b
    best_params = None
    for line in lines:
        params = json.loads(line)
        val = params['Value']
        del params['Value']

        if cmp(val, best_val): 
            best_val = val
            best_params = params
    return best_params
        

def best_test(model_name):
    print(model_name)
    for did in range(1, 4):
        print(f'Test Dataset {did}')
        ans = test(did, model_name, load_and_select(did, model_name))
        df = pd.DataFrame(ans)#, columns=['score','train','inference'])
        dfmean = df.mean()
        dfstd = df.std()
        df = df._append(dfmean, ignore_index=True)
        df = df._append(dfstd, ignore_index=True)
        df.index = list(df.index)[:-2]+['mean','std']
        print(df)

def test_all():
    for model_name in ['xgb', 'lightgbm', 'mlp', 'svm', 'cart']:
        best_test(model_name)





if __name__ == '__main__':
    test_all()
    # did = 3
    # for model_name in ['xgb', 'lightgbm', 'mlp', 'svm', 'cart']:
    #     print(model_name)
    #     ans = test(did, model_name, load_and_select(did, model_name))
    #     df = pd.DataFrame(ans, columns=['score','train','inference'])
    #     df = df._append(df.mean(), ignore_index=True)
    #     df.index = list(df.index)[:-1]+['mean']
    #     print(df)
    # model_name = 'cart'
    # tune_model(2, model_name, 50)
    # tune_model(3, model_name, 50)


