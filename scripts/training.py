import itertools
import json
import os

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


BUCKETS = [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, np.inf),
]
PARAM_GRID = {
    "max_depth": [2,3,4],
    "learning_rate": [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.05],
}


def load_datasets():
    X_train = pd.read_csv("data/train_test_files/X_train.csv")
    X_test = pd.read_csv("data/train_test_files/X_test.csv")
    y_train = pd.read_csv("data/train_test_files/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("data/train_test_files/y_test.csv").iloc[:, 0]

    unified = pd.read_csv("data/train_test_files/unified_train_test.csv")
    test_meta = unified[unified["train_test_team_a"] == "test"][["Gtm_team_a"]].reset_index(drop=True)
    if len(test_meta) != len(X_test):
        raise ValueError("Test metadata row count does not match X_test row count")

    return X_train, X_test, y_train, y_test, test_meta


def prepare_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["site_team_a"] = X_train["site_team_a"].fillna("H").astype("category")
    X_test["site_team_a"] = X_test["site_team_a"].fillna("H").astype("category")
    X_train["Type_team_a"] = X_train["Type_team_a"].astype("category")
    X_test["Type_team_a"] = X_test["Type_team_a"].astype("category")

    numeric_cols = [col for col in X_train.columns if col not in ["site_team_a", "Type_team_a"]]
    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        X_train[col] = np.where(X_train[col].abs() > 1e9, np.nan, X_train[col])
        X_test[col] = np.where(X_test[col].abs() > 1e9, np.nan, X_test[col])

    return X_train, X_test


def compute_bucket_metrics(y_true, preds, gtm_series):
    rows = []
    pred_labels = np.where(preds >= 0.5, 1, 0)

    for bucket_label, start, end in BUCKETS:
        mask = (gtm_series >= start) & (gtm_series <= end)
        games = int(mask.sum())
        if games == 0:
            continue

        bucket_y = y_true[mask]
        bucket_preds = preds[mask]
        bucket_labels = pred_labels[mask]
        rows.append(
            {
                "bucket": bucket_label,
                "games": games,
                "log_loss": log_loss(bucket_y, bucket_preds, labels=[0, 1]),
                "accuracy": accuracy_score(bucket_y, bucket_labels),
            }
        )

    return rows


def fit_model(X_train, X_test, y_train, y_test, params):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        n_estimators=5000,
        early_stopping_rounds=50,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=1)
    return model


def train_candidate(params):
    X_train, X_test, y_train, y_test, test_meta = load_datasets()
    X_train, X_test = prepare_features(X_train, X_test)

    model = fit_model(X_train, X_test, y_train, y_test, params)

    preds = model.predict_proba(X_test)[:, 1]
    pred_labels = np.where(preds >= 0.5, 1, 0)
    best_iteration = model.best_iteration if model.best_iteration is not None else model.n_estimators
    aggregate = {
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "n_estimators": best_iteration,
        "log_loss": log_loss(y_test, preds, labels=[0, 1]),
        "auc": roc_auc_score(y_test, preds),
        "accuracy": accuracy_score(y_test, pred_labels),
    }
    bucket_metrics = compute_bucket_metrics(y_test.to_numpy(), preds, test_meta["Gtm_team_a"].to_numpy())

    return model, aggregate, bucket_metrics, list(X_train.columns), preds


def bucket_score_map(bucket_metrics):
    return {row["bucket"]: row for row in bucket_metrics}


def is_better_candidate(candidate, best):
    if best is None:
        return True
    if candidate["log_loss"] < best["log_loss"]:
        return True
    if candidate["log_loss"] > best["log_loss"]:
        return False
    return candidate["bucket_1_5_log_loss"] < best["bucket_1_5_log_loss"]


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()

    sweep_rows = []
    bucket_rows = []
    best_model = None
    best_summary = None
    best_params = None
    best_feature_columns = None
    best_preds = None

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"params={params}")
        model, aggregate, bucket_metrics, feature_columns, preds = train_candidate(params)
        bucket_map = bucket_score_map(bucket_metrics)
        summary_row = aggregate.copy()
        for bucket_label, _, _ in BUCKETS:
            bucket = bucket_map.get(bucket_label)
            safe_label = bucket_label.replace("+", "plus").replace("-", "_")
            summary_row[f"bucket_{safe_label}_games"] = bucket["games"] if bucket else 0
            summary_row[f"bucket_{safe_label}_log_loss"] = bucket["log_loss"] if bucket else np.nan
            summary_row[f"bucket_{safe_label}_accuracy"] = bucket["accuracy"] if bucket else np.nan

        sweep_rows.append(summary_row)
        for bucket in bucket_metrics:
            bucket_rows.append(
                {
                    "max_depth": params["max_depth"],
                    "learning_rate": params["learning_rate"],
                    **bucket,
                }
            )

        print(f"Score: {aggregate['log_loss']}")
        print(f"Accuracy: {aggregate['accuracy']}")
        print(f'AUC: {aggregate["auc"]}')

        if is_better_candidate(summary_row, best_summary):
            best_model = model
            best_summary = summary_row
            best_params = params
            best_feature_columns = feature_columns
            best_preds = preds

    sweep_df = pd.DataFrame(sweep_rows).sort_values("log_loss")
    sweep_df.to_csv("results/xgb_tuning_results.csv", index=False)
    sweep_df.head(1).to_csv("results/best_params.csv", index=False)

    bucket_df = pd.DataFrame(bucket_rows).sort_values(["max_depth", "learning_rate", "bucket"])
    bucket_df.to_csv("results/game_bucket_metrics.csv", index=False)

    if best_model is None or best_summary is None or best_preds is None:
        raise ValueError("No model was trained")

    best_model.save_model("models/xgb_model.json")
    loaded_model = XGBClassifier()
    loaded_model.load_model("models/xgb_model.json")

    X_train, X_test, y_train, y_test, _ = load_datasets()
    X_train, X_test = prepare_features(X_train, X_test)
    preds_after_save = loaded_model.predict_proba(X_test)[:, 1]
    max_abs_diff = float(np.max(np.abs(best_preds - preds_after_save)))

    with open("models/xgb_model_meta.json", "w") as f:
        json.dump(
            {
                "feature_columns": best_feature_columns,
                "best_params": best_params,
                "best_validation_log_loss": float(best_summary["log_loss"]),
                "xgboost_version": xgboost.__version__,
                "reload_max_abs_diff": max_abs_diff,
            },
            f,
            indent=2,
        )

    print(sweep_df.head())
    print("Best params:")
    print(best_params)
    print("Best validation log loss:")
    print(best_summary["log_loss"])
    print("Saved model:")
    print("models/xgb_model.json")
    print("Reload max abs diff:")
    print(max_abs_diff)
    print("\n\n\n\n\n FINISHED")


if __name__ == "__main__":
    main()
