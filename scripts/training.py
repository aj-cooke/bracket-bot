import argparse
import itertools
import json
import os
from pathlib import Path

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
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05],
}
BENCHMARK_DATA_DIR = Path("data/train_test_files")
PRODUCTION_DATA_DIR = BENCHMARK_DATA_DIR / "production"
PRODUCTION_MODEL_DIR = Path("models/production")
PRODUCTION_RESULTS_DIR = Path("results/production")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "production"], default="benchmark")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--benchmark-params-path", default="results/best_params.csv")
    parser.add_argument("--benchmark-meta-path", default="models/xgb_model_meta.json")
    return parser.parse_args()


def resolve_dataset_dir(args):
    if args.dataset_dir:
        return Path(args.dataset_dir)
    if args.mode == "production":
        return PRODUCTION_DATA_DIR
    return BENCHMARK_DATA_DIR


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_dataset_meta(dataset_dir):
    meta_path = dataset_dir / "dataset_meta.json"
    if meta_path.exists():
        return load_json(meta_path)
    return None


def load_datasets(dataset_dir, require_test):
    X_train = pd.read_csv(dataset_dir / "X_train.csv")
    y_train = pd.read_csv(dataset_dir / "y_train.csv").iloc[:, 0]
    unified = pd.read_csv(dataset_dir / "unified_train_test.csv")

    test_meta = None
    X_test = None
    y_test = None
    if require_test:
        X_test = pd.read_csv(dataset_dir / "X_test.csv")
        y_test = pd.read_csv(dataset_dir / "y_test.csv").iloc[:, 0]
        test_meta = unified[unified["train_test_team_a"] == "test"][["Gtm_team_a"]].reset_index(drop=True)
        if len(test_meta) != len(X_test):
            raise ValueError("Test metadata row count does not match X_test row count")

    return X_train, X_test, y_train, y_test, test_meta


def prepare_features(X_train, X_test=None):
    X_train = X_train.copy()
    X_train["site_team_a"] = X_train["site_team_a"].fillna("H").astype("category")
    X_train["Type_team_a"] = X_train["Type_team_a"].astype("category")

    if X_test is not None:
        X_test = X_test.copy()
        X_test["site_team_a"] = X_test["site_team_a"].fillna("H").astype("category")
        X_test["Type_team_a"] = X_test["Type_team_a"].astype("category")

    numeric_cols = [col for col in X_train.columns if col not in ["site_team_a", "Type_team_a"]]
    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_train[col] = np.where(X_train[col].abs() > 1e9, np.nan, X_train[col])
        if X_test is not None:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
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


def fit_model(X_train, y_train, params, X_eval=None, y_eval=None, n_estimators=5000, use_early_stopping=False):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        n_estimators=n_estimators,
        **params,
    )

    fit_kwargs = {}
    if use_early_stopping:
        model.set_params(early_stopping_rounds=50)
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_eval, y_eval)]
        fit_kwargs["verbose"] = 1

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_candidate(dataset_dir, params):
    X_train, X_test, y_train, y_test, test_meta = load_datasets(dataset_dir, require_test=True)
    X_train, X_test = prepare_features(X_train, X_test)

    model = fit_model(
        X_train,
        y_train,
        params,
        X_eval=X_test,
        y_eval=y_test,
        use_early_stopping=True,
    )

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


def load_locked_benchmark_spec(best_params_path, benchmark_meta_path):
    best_params_row = pd.read_csv(best_params_path).iloc[0]
    benchmark_meta = load_json(benchmark_meta_path)
    return {
        "params": {
            "max_depth": int(best_params_row["max_depth"]),
            "learning_rate": float(best_params_row["learning_rate"]),
        },
        "n_estimators": int(best_params_row["n_estimators"]),
        "feature_columns": benchmark_meta["feature_columns"],
        "benchmark_meta": benchmark_meta,
    }


def verify_feature_columns(X_train, expected_columns):
    actual_columns = list(X_train.columns)
    if actual_columns != expected_columns:
        raise ValueError("Production feature columns do not match locked benchmark feature columns")


def save_production_metadata(model_dir, dataset_meta, locked_spec, reload_max_abs_diff, train_rows):
    payload = {
        "mode": "production",
        "train_rows": int(train_rows),
        "included_years": dataset_meta.get("included_years") if dataset_meta else None,
        "cutoff_date": dataset_meta.get("cutoff_date") if dataset_meta else None,
        "current_season_year": dataset_meta.get("current_season_year") if dataset_meta else None,
        "feature_columns": locked_spec["feature_columns"],
        "best_params": locked_spec["params"],
        "locked_n_estimators": locked_spec["n_estimators"],
        "benchmark_validation_log_loss": locked_spec["benchmark_meta"].get("best_validation_log_loss"),
        "xgboost_version": xgboost.__version__,
        "reload_max_abs_diff": reload_max_abs_diff,
    }
    with open(model_dir / "xgb_model_meta.json", "w") as f:
        json.dump(payload, f, indent=2)


def run_benchmark(args, dataset_dir):
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
        model, aggregate, bucket_metrics, feature_columns, preds = train_candidate(dataset_dir, params)
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

    X_train, X_test, y_train, y_test, _ = load_datasets(dataset_dir, require_test=True)
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


def run_production(args, dataset_dir):
    dataset_meta = load_dataset_meta(dataset_dir)
    if dataset_meta is None:
        raise FileNotFoundError(f"Missing dataset metadata in {dataset_dir}")

    os.makedirs(PRODUCTION_MODEL_DIR, exist_ok=True)
    os.makedirs(PRODUCTION_RESULTS_DIR, exist_ok=True)

    locked_spec = load_locked_benchmark_spec(args.benchmark_params_path, args.benchmark_meta_path)
    X_train, _, y_train, _, _ = load_datasets(dataset_dir, require_test=False)
    verify_feature_columns(X_train, locked_spec["feature_columns"])
    X_train, _ = prepare_features(X_train)

    model = fit_model(
        X_train,
        y_train,
        locked_spec["params"],
        n_estimators=locked_spec["n_estimators"],
        use_early_stopping=False,
    )
    train_preds = model.predict_proba(X_train)[:, 1]
    train_labels = np.where(train_preds >= 0.5, 1, 0)

    train_summary = pd.DataFrame(
        [
            {
                "train_rows": len(X_train),
                "log_loss": log_loss(y_train, train_preds, labels=[0, 1]),
                "auc": roc_auc_score(y_train, train_preds),
                "accuracy": accuracy_score(y_train, train_labels),
                "max_depth": locked_spec["params"]["max_depth"],
                "learning_rate": locked_spec["params"]["learning_rate"],
                "n_estimators": locked_spec["n_estimators"],
                "cutoff_date": dataset_meta.get("cutoff_date"),
            }
        ]
    )
    train_summary.to_csv(PRODUCTION_RESULTS_DIR / "train_metrics.csv", index=False)

    model_path = PRODUCTION_MODEL_DIR / "xgb_model.json"
    model.save_model(model_path)
    loaded_model = XGBClassifier()
    loaded_model.load_model(model_path)
    preds_after_save = loaded_model.predict_proba(X_train)[:, 1]
    max_abs_diff = float(np.max(np.abs(train_preds - preds_after_save)))
    save_production_metadata(
        PRODUCTION_MODEL_DIR,
        dataset_meta,
        locked_spec,
        max_abs_diff,
        len(X_train),
    )

    print(train_summary)
    print("Saved production model:")
    print(model_path)
    print("Reload max abs diff:")
    print(max_abs_diff)


def main():
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args)
    if args.mode == "production":
        run_production(args, dataset_dir)
        return
    run_benchmark(args, dataset_dir)


if __name__ == "__main__":
    main()
