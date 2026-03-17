import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from season_priors import build_all_season_features, load_season_features


warnings.filterwarnings("ignore")

FAST_HALF_LIFE_DAYS = 14.0
SLOW_HALF_LIFE_DAYS = 45.0
RIDGE_ALPHA = 1.0
BLEND_SCHEDULE = os.getenv("BLEND_SCHEDULE", "exp_4")
FACTOR_SPECS = [
    ("efg", "opp_efg"),
    ("tov_rate", "opp_tov_rate"),
    ("orb_rate", "opp_orb_rate"),
    ("ftr", "opp_ftr"),
]
PRIOR_METRIC_COLS = [
    "off_ppp",
    "def_ppp",
    "margin_ppp",
    "pace",
    "efg",
    "opp_efg",
    "tov_rate",
    "opp_tov_rate",
    "orb_rate",
    "opp_orb_rate",
    "ftr",
    "opp_ftr",
    "threepar",
    "opp_threepar",
    "three_pct",
    "opp_three_pct",
    "ft_pct",
    "opp_ft_pct",
]
BLEND_BASES = ["ewm_fast", "ewm_slow", "reg_adj_fast", "reg_adj_slow"]


def discover_years():
    years = []
    for path in sorted(Path("data/years").glob("games_*.csv")):
        try:
            years.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not years:
        raise FileNotFoundError("No cleaned game files found in data/years")
    return years


def exponential_weights(date_series, current_date, half_life_days):
    days_ago = (current_date - date_series).dt.days.clip(lower=0)
    return np.exp(-np.log(2.0) * days_ago / half_life_days)


def build_design_matrix(history, teams):
    team_codes = pd.Categorical(history["team"], categories=teams).codes
    opp_codes = pd.Categorical(history["opp_slug"], categories=teams).codes

    team_mat = sparse.csr_matrix(
        (np.ones(len(history)), (np.arange(len(history)), team_codes)),
        shape=(len(history), len(teams)),
    )
    opp_mat = sparse.csr_matrix(
        (np.ones(len(history)), (np.arange(len(history)), opp_codes)),
        shape=(len(history), len(teams)),
    )
    return sparse.hstack([team_mat, opp_mat], format="csr")


def fit_team_rating(history, teams, current_date, half_life_days, response_col):
    history = history[history[response_col].notna()].copy()
    if history.empty:
        return np.full(len(teams), np.nan)

    X = build_design_matrix(history, teams)
    weights = exponential_weights(history["Date"], current_date, half_life_days)
    y = history[response_col].to_numpy()

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr")
    model.fit(X, y, sample_weight=weights)

    team_effect = model.coef_[: len(teams)]
    team_weights = history.groupby("team")["weight_tmp"].sum().reindex(teams, fill_value=0.0).to_numpy()
    if team_weights.sum() > 0:
        team_effect = team_effect - np.average(team_effect, weights=team_weights)

    return model.intercept_ + team_effect


def fit_tempo_rating(history, teams, current_date, half_life_days):
    history = history[history["pace"].notna()].copy()
    if history.empty:
        return np.full(len(teams), np.nan)

    X = build_design_matrix(history, teams)
    weights = exponential_weights(history["Date"], current_date, half_life_days)
    y = history["pace"].to_numpy()

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr")
    model.fit(X, y, sample_weight=weights)

    team_effect = model.coef_[: len(teams)]
    opp_effect = model.coef_[len(teams) :]
    combined_effect = 0.5 * (team_effect + opp_effect)

    team_weights = history.groupby("team")["weight_tmp"].sum().reindex(teams, fill_value=0.0).to_numpy()
    opp_weights = history.groupby("opp_slug")["weight_tmp"].sum().reindex(teams, fill_value=0.0).to_numpy()
    combined_weights = team_weights + opp_weights
    if combined_weights.sum() > 0:
        combined_effect = combined_effect - np.average(combined_effect, weights=combined_weights)

    return model.intercept_ + combined_effect


def fit_snapshot(history, teams, current_date, half_life_days):
    history = history.copy()
    history["weight_tmp"] = exponential_weights(history["Date"], current_date, half_life_days)

    snapshot = pd.DataFrame({"team": teams, "rating_date": current_date})
    snapshot["off_ppp"] = fit_team_rating(history, teams, current_date, half_life_days, "off_ppp")
    snapshot["def_ppp"] = fit_team_rating(history, teams, current_date, half_life_days, "def_ppp")
    snapshot["margin_ppp"] = snapshot["off_ppp"] - snapshot["def_ppp"]
    snapshot["pace"] = fit_tempo_rating(history, teams, current_date, half_life_days)

    for off_col, def_col in FACTOR_SPECS:
        snapshot[off_col] = fit_team_rating(history, teams, current_date, half_life_days, off_col)
        snapshot[def_col] = fit_team_rating(history, teams, current_date, half_life_days, def_col)

    return snapshot


def build_snapshots(df, teams, half_life_days, prefix):
    snapshots = []
    for current_date in sorted(df["Date"].unique()):
        history = df[df["Date"] < current_date]
        if history.empty:
            snapshot = pd.DataFrame({"team": teams, "rating_date": current_date})
            snapshot["off_ppp"] = np.nan
            snapshot["def_ppp"] = np.nan
            snapshot["margin_ppp"] = np.nan
            snapshot["pace"] = np.nan
            for off_col, def_col in FACTOR_SPECS:
                snapshot[off_col] = np.nan
                snapshot[def_col] = np.nan
        else:
            snapshot = fit_snapshot(history, teams, current_date, half_life_days)
            fill_cols = ["off_ppp", "def_ppp", "pace"]
            for off_col, def_col in FACTOR_SPECS:
                fill_cols.extend([off_col, def_col])

            for col in fill_cols:
                snapshot[col] = snapshot[col].fillna(history[col].mean())

            snapshot["margin_ppp"] = snapshot["off_ppp"] - snapshot["def_ppp"]

        rename_map = {
            "rating_date": "Date",
            "off_ppp": f"{prefix}_off_ppp",
            "def_ppp": f"{prefix}_def_ppp",
            "margin_ppp": f"{prefix}_margin_ppp",
            "pace": f"{prefix}_pace",
        }
        for off_col, def_col in FACTOR_SPECS:
            rename_map[off_col] = f"{prefix}_{off_col}"
            rename_map[def_col] = f"{prefix}_{def_col}"

        snapshot = snapshot.rename(columns=rename_map)
        snapshots.append(snapshot)

    return pd.concat(snapshots, ignore_index=True)


def add_previous_season_priors(df, year):
    priors = load_season_features(year - 1).drop(columns=["season"], errors="ignore")
    df = df.merge(priors, on="team", how="left")
    prior_games = pd.to_numeric(df["Gtm"], errors="coerce").fillna(1.0) - 1.0
    df["prior_weight"] = compute_prior_weight(prior_games)
    df["prior_blend_schedule"] = BLEND_SCHEDULE
    return df


def compute_prior_weight(prior_games):
    if BLEND_SCHEDULE == "linear_5":
        return np.clip(1.0 - (prior_games / 5.0), 0.0, 1.0)
    if BLEND_SCHEDULE == "linear_8":
        return np.clip(1.0 - (prior_games / 8.0), 0.0, 1.0)
    if BLEND_SCHEDULE == "linear_10":
        return np.clip(1.0 - (prior_games / 10.0), 0.0, 1.0)
    if BLEND_SCHEDULE == "linear_12":
        return np.clip(1.0 - (prior_games / 12.0), 0.0, 1.0)
    if BLEND_SCHEDULE == "linear_15":
        return np.clip(1.0 - (prior_games / 15.0), 0.0, 1.0)
    if BLEND_SCHEDULE == "exp_4":
        return np.exp(-prior_games / 4.0)
    raise ValueError(f"Unsupported BLEND_SCHEDULE: {BLEND_SCHEDULE}")


def get_prior_metric(col):
    if col.startswith("ewm_"):
        metric = col.split("_", 2)[2]
    elif col.startswith("reg_adj_"):
        metric = col.split("_", 3)[3]
    else:
        return None

    prior_col = f"prev_{metric}"
    if metric in PRIOR_METRIC_COLS:
        return prior_col
    return None


def blend_state_column(current, prior, weight):
    result = current.copy()
    both = current.notna() & prior.notna()
    result = result.where(~both, weight * prior + (1.0 - weight) * current)
    result = result.where(current.notna() | prior.isna(), prior)
    return result


def add_blended_state_features(df):
    state_cols = [col for col in df.columns if col.startswith("ewm_") or col.startswith("reg_adj_")]

    for col in state_cols:
        prior_col = get_prior_metric(col)
        blend_col = f"blend_{col}"
        if prior_col is None or prior_col not in df.columns:
            df[blend_col] = df[col]
            continue
        df[blend_col] = blend_state_column(df[col], df[prior_col], df["prior_weight"])

    for base in BLEND_BASES:
        off_col = f"blend_{base}_off_ppp"
        def_col = f"blend_{base}_def_ppp"
        margin_col = f"blend_{base}_margin_ppp"
        if off_col in df.columns and def_col in df.columns:
            df[margin_col] = df[off_col] - df[def_col]

    return df


build_all_season_features()

for year in discover_years():
    df = pd.read_csv(f"data/years/games_{year}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = add_previous_season_priors(df, year)
    teams = sorted(set(df["team"].dropna()) | set(df["opp_slug"].dropna()))

    fast = build_snapshots(df, teams, FAST_HALF_LIFE_DAYS, "reg_adj_fast")
    slow = build_snapshots(df, teams, SLOW_HALF_LIFE_DAYS, "reg_adj_slow")

    df = df.merge(fast, on=["team", "Date"], how="left")
    df = df.merge(slow, on=["team", "Date"], how="left")
    df = add_blended_state_features(df)

    df = df.sort_values(["Date", "team", "Gtm"]).reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(f"data/years/all_games_{year}_with_sos_features.csv", index=False)
