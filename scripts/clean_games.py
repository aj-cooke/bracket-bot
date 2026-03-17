import json
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

YEARS = np.arange(2021, 2027, 1)
FAST_SPAN = 5
SLOW_SPAN = 15

with open("data/schools.json", "r") as file:
    SCHOOLS = json.load(file)["schools"]

RAW_BOX_COLS = [
    "score_Rslt",
    "score_Tm",
    "score_Opp",
    "FG",
    "FGA",
    "3P",
    "3PA",
    "2P",
    "2PA",
    "FT",
    "FTA",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "opp_FG",
    "opp_FGA",
    "opp_3P",
    "opp_3PA",
    "opp_2P",
    "opp_2PA",
    "opp_FT",
    "opp_FTA",
    "opp_ORB",
    "opp_DRB",
    "opp_TRB",
    "opp_AST",
    "opp_STL",
    "opp_BLK",
    "opp_TOV",
    "opp_PF",
]

GAME_FEATURES = [
    "win",
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
    "two_pct",
    "opp_two_pct",
    "three_pct",
    "opp_three_pct",
    "ft_pct",
    "opp_ft_pct",
    "ast_rate",
    "opp_ast_rate",
    "stl_rate",
    "opp_stl_rate",
    "blk_rate",
    "opp_blk_rate",
]


def clean_game_df(df):
    col_prefixes = []
    for col in df.columns:
        if col.startswith("Opponent"):
            col_prefixes.append("opp_")
        elif col.startswith("Score"):
            col_prefixes.append("score_")
        elif col.startswith("Defensive"):
            col_prefixes.append("opp_")
        else:
            col_prefixes.append("")

    cols = list(df.iloc[0, :])
    df.columns = [col_prefixes[i] + cols[i] for i in range(len(cols))]
    df.columns = ["site" if col.startswith("Unnamed") else col for col in df.columns]
    df = df[(df["Rk"] != "Rk") & (~df["Rk"].isna())]
    del df["Rk"]
    df = df.set_index("Gtm")
    return df


def safe_divide(num, den):
    return np.where(pd.to_numeric(den, errors="coerce") == 0, np.nan, num / den)


def add_game_features(df):
    df["site"] = df["site"].fillna("H")
    df["Type"] = np.where(
        ~df["Type"].isin(["REG (Conf)", "REG (Non-Conf)", "CTOURN"]),
        "PT",
        df["Type"],
    )
    df["win"] = np.where(df["score_Rslt"] == "W", 1.0, 0.0)

    team_poss = df["FGA"] - df["ORB"] + df["TOV"] + 0.475 * df["FTA"]
    opp_poss = df["opp_FGA"] - df["opp_ORB"] + df["opp_TOV"] + 0.475 * df["opp_FTA"]
    df["poss"] = (team_poss + opp_poss) / 2.0
    df["pace"] = df["poss"]

    df["off_ppp"] = safe_divide(df["score_Tm"], df["poss"])
    df["def_ppp"] = safe_divide(df["score_Opp"], df["poss"])
    df["margin_ppp"] = df["off_ppp"] - df["def_ppp"]

    df["efg"] = safe_divide(df["FG"] + 0.5 * df["3P"], df["FGA"])
    df["opp_efg"] = safe_divide(df["opp_FG"] + 0.5 * df["opp_3P"], df["opp_FGA"])
    df["tov_rate"] = safe_divide(df["TOV"], df["poss"])
    df["opp_tov_rate"] = safe_divide(df["opp_TOV"], df["poss"])
    df["orb_rate"] = safe_divide(df["ORB"], df["ORB"] + df["opp_DRB"])
    df["opp_orb_rate"] = safe_divide(df["opp_ORB"], df["opp_ORB"] + df["DRB"])
    df["ftr"] = safe_divide(df["FTA"], df["FGA"])
    df["opp_ftr"] = safe_divide(df["opp_FTA"], df["opp_FGA"])
    df["threepar"] = safe_divide(df["3PA"], df["FGA"])
    df["opp_threepar"] = safe_divide(df["opp_3PA"], df["opp_FGA"])
    df["two_pct"] = safe_divide(df["2P"], df["2PA"])
    df["opp_two_pct"] = safe_divide(df["opp_2P"], df["opp_2PA"])
    df["three_pct"] = safe_divide(df["3P"], df["3PA"])
    df["opp_three_pct"] = safe_divide(df["opp_3P"], df["opp_3PA"])
    df["ft_pct"] = safe_divide(df["FT"], df["FTA"])
    df["opp_ft_pct"] = safe_divide(df["opp_FT"], df["opp_FTA"])
    df["ast_rate"] = safe_divide(df["AST"], df["FG"])
    df["opp_ast_rate"] = safe_divide(df["opp_AST"], df["opp_FG"])
    df["stl_rate"] = safe_divide(df["STL"], df["poss"])
    df["opp_stl_rate"] = safe_divide(df["opp_STL"], df["poss"])
    df["blk_rate"] = safe_divide(df["BLK"], df["opp_2PA"])
    df["opp_blk_rate"] = safe_divide(df["opp_BLK"], df["2PA"])

    for col in GAME_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_ewm_features(df):
    df = df.sort_values(["Date", "Gtm"]).copy()
    for col in GAME_FEATURES:
        df[f"ewm_fast_{col}"] = (
            df.groupby("team")[col].transform(
                lambda s: s.shift(1).ewm(span=FAST_SPAN, adjust=False, min_periods=1).mean()
            )
        )
        df[f"ewm_slow_{col}"] = (
            df.groupby("team")[col].transform(
                lambda s: s.shift(1).ewm(span=SLOW_SPAN, adjust=False, min_periods=1).mean()
            )
        )
    return df


for year in YEARS:
    df_year = pd.DataFrame()
    for school in SCHOOLS:
        try:
            df_basic = pd.read_csv(f"data/games/{school}_{year}_basic.csv")
            df_advanced = pd.read_csv(f"data/games/{school}_{year}_advanced.csv")
        except Exception:
            print(f"no games for {school} in {year}")
            continue

        df_basic = clean_game_df(df_basic)
        df_advanced = clean_game_df(df_advanced)
        df = df_basic.join(df_advanced, rsuffix="_drop")
        df = df.filter(regex="^(?!.*_drop$)")

        keep_cols = ["Date", "site", "Type", "Opp"] + RAW_BOX_COLS
        df = df[keep_cols].copy()

        for col in RAW_BOX_COLS:
            if col == "score_Rslt":
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = add_game_features(df)
        df = df.reset_index()
        df["team"] = school
        df["opp_slug"] = (
            df["Opp"]
            .str.lower()
            .str.replace("@", " ", regex=False)
            .str.replace(r"[^\w\s-]", "", regex=True)
            .str.replace(r"\s+", "-", regex=True)
            .str.replace(r"-+", "-", regex=True)
            .str.strip("-")
        )

        df_year = pd.concat([df_year, df], axis=0, ignore_index=True)
        print(f"{year} {school} complete")

    if df_year.empty:
        print(f"{year} has no games")
        print("\n\n")
        continue

    df_year["Date"] = pd.to_datetime(df_year["Date"])
    df_year = add_ewm_features(df_year)
    df_year = df_year.sort_values(["Date", "team", "Gtm"]).reset_index(drop=True)
    df_year["Date"] = df_year["Date"].dt.strftime("%Y-%m-%d")
    df_year.to_csv(f"data/years/games_{year}.csv", index=False)
    print(f"{year} complete")
    print("\n\n")
