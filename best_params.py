import pandas as pd
import os
from main import load_and_select

def best_params():
    save_dir = './params'
    os.makedirs(save_dir, exist_ok=True)
    for data_id in range(1,4):
        print('-'*100)
        print('Dataset: ', data_id)
        for idx, model_name in enumerate(['lightgbm', 'xgb', 'mlp', 'svm', 'cart']):
            best_dict = load_and_select(data_id, model_name)
            df = pd.DataFrame({model_name:best_dict})
            df.to_csv(f'{save_dir}/dataset{data_id}-m{idx}-{model_name}.csv')
            
            print(model_name)

best_params()