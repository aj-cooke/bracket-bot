import numpy as np
import pandas as pd


YEARS = np.arange(2021, 2025, 1)
TEST_YEAR = 2025
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


df = pd.read_csv(f"data/years/all_games_{TEST_YEAR}_with_sos_features.csv")
df["train_test"] = "test"

for year in YEARS:
    cur = pd.read_csv(f"data/years/all_games_{year}_with_sos_features.csv")
    cur["train_test"] = "train"
    df = pd.concat([df, cur], axis=0, ignore_index=True)

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

XCOLS = ["site_team_a", "Type_team_a"] + sorted(
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

df_train = df[df["train_test_team_a"] == "train"].copy()
df_test = df[df["train_test_team_a"] == "test"].copy()
X_train = df_train[XCOLS]
X_test = df_test[XCOLS]
y_train = df_train[[LABEL_COL]]
y_test = df_test[[LABEL_COL]]

df.to_csv("data/train_test_files/unified_train_test.csv", index=False)
X_train.to_csv("data/train_test_files/X_train.csv", index=False)
X_test.to_csv("data/train_test_files/X_test.csv", index=False)
y_train.to_csv("data/train_test_files/y_train.csv", index=False)
y_test.to_csv("data/train_test_files/y_test.csv", index=False)
pd.DataFrame({"column": XCOLS}).to_csv("data/xcols.csv", index=False)
