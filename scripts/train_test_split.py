import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COL = "win_team_a"
MIN_GTM = 0
FACTOR_COLS = ["efg", "tov_rate", "orb_rate", "ftr"]
DIRECT_PRIOR_COLS = [
    "prev_srs",
    "prev_sos",
    "prev_win_pct",
    "prev_threepar",
    "prev_opp_threepar",
    "prev_three_pct",
    "prev_opp_three_pct",
    "prev_ft_pct",
    "prev_opp_ft_pct",
]
BENCHMARK_DIR = Path("data/train_test_files")
PRODUCTION_DIR = BENCHMARK_DIR / "production"
YEAR_FILE_TEMPLATE = "all_games_{year}_with_sos_features.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "production"], default="benchmark")
    parser.add_argument("--train-start-year", type=int, default=2021)
    parser.add_argument("--benchmark-test-year", type=int, default=2025)
    parser.add_argument("--current-season-year", type=int)
    parser.add_argument("--cutoff-date")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def discover_available_years():
    years = []
    for path in sorted(Path("data/years").glob("all_games_*_with_sos_features.csv")):
        try:
            years.append(int(path.stem.split("_")[2]))
        except (IndexError, ValueError):
            continue
    if not years:
        raise FileNotFoundError("No season feature files found in data/years")
    return years


def get_output_dir(args):
    if args.output_dir:
        return Path(args.output_dir)
    if args.mode == "production":
        return PRODUCTION_DIR
    return BENCHMARK_DIR


def load_year_frame(year):
    path = Path("data/years") / YEAR_FILE_TEMPLATE.format(year=year)
    if not path.exists():
        raise FileNotFoundError(f"Missing season feature file: {path}")
    df = pd.read_csv(path)
    df["source_season"] = year
    return df


def benchmark_source_years(args):
    available_years = discover_available_years()
    if args.benchmark_test_year not in available_years:
        raise ValueError(f"Benchmark test year {args.benchmark_test_year} is not available")

    train_years = [year for year in available_years if args.train_start_year <= year < args.benchmark_test_year]
    if not train_years:
        raise ValueError("Benchmark mode found no train years before the test year")

    return train_years, args.benchmark_test_year


def production_source_years(args):
    available_years = discover_available_years()
    eligible_years = [year for year in available_years if year >= args.train_start_year]
    if args.current_season_year is not None:
        eligible_years = [year for year in eligible_years if year <= args.current_season_year]
    if not eligible_years:
        raise ValueError("Production mode found no eligible season feature files")
    return eligible_years


def load_source_frames(args):
    if args.mode == "benchmark":
        train_years, test_year = benchmark_source_years(args)
        frames = []
        for year in train_years:
            cur = load_year_frame(year)
            cur["train_test"] = "train"
            frames.append(cur)
        test_df = load_year_frame(test_year)
        test_df["train_test"] = "test"
        frames.append(test_df)
        return pd.concat(frames, axis=0, ignore_index=True), train_years, test_year

    production_years = production_source_years(args)
    frames = []
    for year in production_years:
        cur = load_year_frame(year)
        if args.cutoff_date and year == production_years[-1]:
            cur["Date"] = pd.to_datetime(cur["Date"])
            cutoff = pd.Timestamp(args.cutoff_date)
            cur = cur[cur["Date"] <= cutoff].copy()
            cur["Date"] = cur["Date"].dt.strftime("%Y-%m-%d")
        cur["train_test"] = "train"
        frames.append(cur)

    return pd.concat(frames, axis=0, ignore_index=True), production_years, None


def build_matchup_frame(df):
    state_cols = [
        col
        for col in df.columns
        if col.startswith("blend_ewm_fast_")
        or col.startswith("blend_ewm_slow_")
        or col.startswith("blend_reg_adj_")
    ]

    df = df[df["Gtm"] > MIN_GTM].copy()
    df = pd.merge(
        df,
        df,
        how="inner",
        left_on=["opp_slug", "Date"],
        right_on=["team", "Date"],
        suffixes=("_team_a", "_team_b"),
    )

    for col in state_cols:
        a_col = f"{col}_team_a"
        b_col = f"{col}_team_b"
        df[f"matchup_delta_{col}"] = df[a_col] - df[b_col]
        if col.endswith("pace"):
            df[f"matchup_mean_{col}"] = (df[a_col] + df[b_col]) / 2.0

    for col in DIRECT_PRIOR_COLS:
        df[f"matchup_delta_{col}"] = df[f"{col}_team_a"] - df[f"{col}_team_b"]

    df["matchup_mean_blend_ewm_fast_threepar"] = (
        df["blend_ewm_fast_threepar_team_a"] + df["blend_ewm_fast_threepar_team_b"]
    ) / 2.0
    df["matchup_mean_blend_ewm_slow_threepar"] = (
        df["blend_ewm_slow_threepar_team_a"] + df["blend_ewm_slow_threepar_team_b"]
    ) / 2.0
    df["matchup_3par_edge_fast"] = (
        df["blend_ewm_fast_threepar_team_a"] - df["blend_ewm_fast_opp_threepar_team_b"]
    )
    df["matchup_3par_edge_slow"] = (
        df["blend_ewm_slow_threepar_team_a"] - df["blend_ewm_slow_opp_threepar_team_b"]
    )
    df["matchup_opp_3par_edge_fast"] = (
        df["blend_ewm_fast_opp_threepar_team_a"] - df["blend_ewm_fast_threepar_team_b"]
    )
    df["matchup_opp_3par_edge_slow"] = (
        df["blend_ewm_slow_opp_threepar_team_a"] - df["blend_ewm_slow_threepar_team_b"]
    )

    df["matchup_off_edge_fast"] = (
        df["blend_ewm_fast_off_ppp_team_a"] - df["blend_ewm_fast_def_ppp_team_b"]
    )
    df["matchup_off_edge_slow"] = (
        df["blend_ewm_slow_off_ppp_team_a"] - df["blend_ewm_slow_def_ppp_team_b"]
    )
    df["matchup_def_edge_fast"] = (
        df["blend_ewm_fast_def_ppp_team_a"] - df["blend_ewm_fast_off_ppp_team_b"]
    )
    df["matchup_def_edge_slow"] = (
        df["blend_ewm_slow_def_ppp_team_a"] - df["blend_ewm_slow_off_ppp_team_b"]
    )
    df["matchup_adj_edge_fast"] = (
        df["blend_reg_adj_fast_margin_ppp_team_a"] - df["blend_reg_adj_fast_margin_ppp_team_b"]
    )
    df["matchup_adj_edge_slow"] = (
        df["blend_reg_adj_slow_margin_ppp_team_a"] - df["blend_reg_adj_slow_margin_ppp_team_b"]
    )
    df["matchup_adj_off_edge_fast"] = (
        df["blend_reg_adj_fast_off_ppp_team_a"] - df["blend_reg_adj_fast_def_ppp_team_b"]
    )
    df["matchup_adj_off_edge_slow"] = (
        df["blend_reg_adj_slow_off_ppp_team_a"] - df["blend_reg_adj_slow_def_ppp_team_b"]
    )
    df["matchup_adj_def_edge_fast"] = (
        df["blend_reg_adj_fast_def_ppp_team_a"] - df["blend_reg_adj_fast_off_ppp_team_b"]
    )
    df["matchup_adj_def_edge_slow"] = (
        df["blend_reg_adj_slow_def_ppp_team_a"] - df["blend_reg_adj_slow_off_ppp_team_b"]
    )

    for horizon in ["fast", "slow"]:
        for factor in FACTOR_COLS:
            df[f"matchup_adj_{factor}_edge_{horizon}"] = (
                df[f"blend_reg_adj_{horizon}_{factor}_team_a"]
                - df[f"blend_reg_adj_{horizon}_opp_{factor}_team_b"]
            )
            df[f"matchup_adj_opp_{factor}_edge_{horizon}"] = (
                df[f"blend_reg_adj_{horizon}_opp_{factor}_team_a"]
                - df[f"blend_reg_adj_{horizon}_{factor}_team_b"]
            )

    return df


def feature_columns(df):
    return ["site_team_a", "Type_team_a"] + sorted(
        [
            col
            for col in df.columns
            if col.startswith("matchup_delta_")
            or col.startswith("matchup_mean_")
            or col.startswith("matchup_off_edge_")
            or col.startswith("matchup_def_edge_")
            or col.startswith("matchup_3par_edge_")
            or col.startswith("matchup_opp_3par_edge_")
            or col.startswith("matchup_adj_")
        ]
    )


def write_csv_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def write_json_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(path)


def build_metadata(args, output_dir, df, xcols, included_years, test_year):
    metadata = {
        "mode": args.mode,
        "output_dir": str(output_dir),
        "train_start_year": args.train_start_year,
        "included_years": included_years,
        "feature_count": len(xcols),
        "rows_unified": int(len(df)),
    }
    if args.mode == "benchmark":
        metadata["benchmark_test_year"] = test_year
        metadata["train_rows"] = int((df["train_test_team_a"] == "train").sum())
        metadata["test_rows"] = int((df["train_test_team_a"] == "test").sum())
    else:
        metadata["current_season_year"] = args.current_season_year
        metadata["cutoff_date"] = args.cutoff_date
        metadata["train_rows"] = int((df["train_test_team_a"] == "train").sum())
        metadata["test_rows"] = 0
    return metadata


def main():
    args = parse_args()
    output_dir = get_output_dir(args)

    source_df, included_years, test_year = load_source_frames(args)
    df = build_matchup_frame(source_df)
    xcols = feature_columns(df)

    df_train = df[df["train_test_team_a"] == "train"].copy()
    X_train = df_train[xcols]
    y_train = df_train[[LABEL_COL]]

    files_to_write = {
        "unified_train_test.csv": df,
        "X_train.csv": X_train,
        "y_train.csv": y_train,
    }

    if args.mode == "benchmark":
        df_test = df[df["train_test_team_a"] == "test"].copy()
        files_to_write["X_test.csv"] = df_test[xcols]
        files_to_write["y_test.csv"] = df_test[[LABEL_COL]]

    for file_name, frame in files_to_write.items():
        write_csv_atomic(frame, output_dir / file_name)

    xcols_path = Path("data/xcols.csv") if args.mode == "benchmark" else output_dir / "xcols.csv"
    write_csv_atomic(pd.DataFrame({"column": xcols}), xcols_path)

    metadata = build_metadata(args, output_dir, df, xcols, included_years, test_year)
    write_json_atomic(metadata, output_dir / "dataset_meta.json")


if __name__ == "__main__":
    main()
