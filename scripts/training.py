import pandas as pd
import numpy as np
import itertools
from xgboost import XGBClassifier
from sklearn.metrics import log_loss


X_train = pd.read_csv('Documents/bracket-bot/data/train_test_files/X_train.csv')
print([x for x in X_train.columns if x in ['team', 'opp_slug']])
print(12/0)
X_test = pd.read_csv('Documents/bracket-bot/data/train_test_files/X_test.csv')
y_train = pd.read_csv('Documents/bracket-bot/data/train_test_files/y_train.csv')
y_test = pd.read_csv('Documents/bracket-bot/data/train_test_files/y_test.csv')

param_grid = {
    'n_estimators': [100],# [100, 200, 400],
    'max_depth': [3], #, 5],
    'learning_rate': [0.1], #[0.03, 0.1],
    #'subsample': [0.8, 1.0],
    #'colsample_bytree': [0.8, 1.0],
    #'min_child_weight': [1, 5]
}

keys = param_grid.keys()
values = param_grid.values()

results = []

best_score = float("inf")
best_model = None
best_params = None

for combo in itertools.product(*values):
    print(combo)

    params = dict(zip(keys, combo))

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        **params
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    score = log_loss(y_test, preds)
    results.append({**params, "log_loss": score})

    if score < best_score:
        best_score = score
        best_model = model
        best_params = params

results_df = pd.DataFrame(results).sort_values("log_loss")

print("Best params:")
print(best_params)

print("Best validation log loss:")
print(best_score)