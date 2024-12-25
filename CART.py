import numpy as np
import pandas as pd
from collections import Counter

class CART:
    def __init__(self, min_samples_split=2, max_depth=50, cart_type='regression'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None
        if cart_type == 'regression': 
            self.regress = True
            self.loss_func = self.mse
        else: 
            self.regress = False
            self.loss_func = self.gini

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.tree = self.build_cart_tree(X, y, depth=0)

    def predict(self, X):
        X = pd.DataFrame(X)
        return X.apply(lambda row: self.predict_one(row, self.tree), axis=1).values

    def mse(self, y):
        if len(y) == 0: return 0
        return np.mean((y - np.mean(y)) ** 2)

    def gini(self, y):
        if len(y) == 0: return 0
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return 1 - sum(p ** 2 for p in probabilities)

    def find_best_split(self, X, y):
        best_feature, best_split, best_metric = None, None, float('inf')
        for feature in X.columns:
            values = X[feature].sort_values().unique()
            for i in range(len(values) - 1):
                split = (values[i] + values[i + 1]) / 2
                left_mask = X[feature] <= split
                right_mask = X[feature] > split

                left_y, right_y = y[left_mask], y[right_mask]
                metric = (len(left_y) * self.loss_func(left_y) + len(right_y) * self.loss_func(right_y)) / len(y)

                if metric < best_metric:
                    best_feature, best_split, best_metric = feature, split, metric

        return best_feature, best_split, best_metric

    def build_cart_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            if self.regress: return {'value': np.mean(y)}
            else: return {'value': y.value_counts().index[0]}

        feature, split, _ = self.find_best_split(X, y)
        if feature is None: 
            if self.regress: return {'value': np.mean(y)}
            else: return {'value': y.value_counts().index[0]}

        # split tree
        left_mask = X[feature] <= split
        right_mask = X[feature] > split

        left_tree = self.build_cart_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_cart_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': feature,
            'split': split,
            'left': left_tree,
            'right': right_tree
        }

    def predict_one(self, row, tree):
        if 'value' in tree:  return tree['value']
        if row[tree['feature']] <= tree['split']: return self.predict_one(row, tree['left'])
        else: return self.predict_one(row, tree['right'])



# if __name__ == '__main__':
#     from model import load_data
#     X_train, y_train, X_test, y_test = load_data(3)
#     cart = CART(cart_type='classification', max_depth=5)
#     cart.fit(X_train, y_train)
#     pred = cart.predict(X_test)
#     from sklearn.metrics import f1_score
#     s = f1_score(pred, y_test)
#     print(s)