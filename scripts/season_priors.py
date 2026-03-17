import re
from pathlib import Path

import pandas as pd


SEASONS_DIR = Path("data/seasons")
TEAM_ALIASES = {
    "bowling-green": "bowling-green-state",
    "east-texas-am": "texas-am-commerce",
    "fdu": "fairleigh-dickinson",
    "houston-christian": "houston-baptist",
    "iu-indy": "iupui",
    "kansas-city": "missouri-kansas-city",
    "little-rock": "arkansas-little-rock",
    "louisiana": "louisiana-lafayette",
    "nc-state": "north-carolina-state",
    "omaha": "nebraska-omaha",
    "purdue-fort-wayne": "ipfw",
    "sam-houston": "sam-houston-state",
    "siu-edwardsville": "southern-illinois-edwardsville",
    "st-thomas": "st-thomas-mn",
    "tcu": "texas-christian",
    "texas-rio-grande-valley": "texas-pan-american",
    "the-citadel": "citadel",
    "uab": "alabama-birmingham",
    "uc-davis": "california-davis",
    "uc-irvine": "california-irvine",
    "uc-riverside": "california-riverside",
    "uc-san-diego": "california-san-diego",
    "uc-santa-barbara": "california-santa-barbara",
    "ucf": "central-florida",
    "unc-asheville": "north-carolina-asheville",
    "unc-greensboro": "north-carolina-greensboro",
    "unc-wilmington": "north-carolina-wilmington",
    "ut-arlington": "texas-arlington",
    "utah-tech": "dixie-state",
    "utep": "texas-el-paso",
    "utsa": "texas-san-antonio",
    "vmi": "virginia-military-institute",
}


def _flatten_columns(df):
    flat_cols = []
    for top, bottom in df.columns:
        top = str(top)
        bottom = str(bottom)
        if "Unnamed" in bottom:
            flat_cols.append(top)
        else:
            flat_cols.append(bottom)
    df = df.copy()
    df.columns = flat_cols
    return df


def _normalize_school_name(name):
    name = str(name).replace("\xa0", " ")
    name = re.sub(r"\bNCAA\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _slugify_school(name):
    name = _normalize_school_name(name).lower()
    name = name.replace("@", " ")
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")
    return TEAM_ALIASES.get(name, name)


def _load_table(path, keep_cols):
    df = pd.read_csv(path, header=[0, 1])
    df = _flatten_columns(df)
    df = df[["School", *keep_cols]].copy()
    df["School"] = df["School"].map(_normalize_school_name)
    df["team"] = df["School"].map(_slugify_school)
    df = df[~df["team"].isin(["", "nan", "school"])].copy()

    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.drop_duplicates(subset=["team"]).reset_index(drop=True)


def build_season_features_for_year(year):
    basic = _load_table(
        SEASONS_DIR / f"basic_{year}.csv",
        ["W-L%", "SRS", "SOS"],
    ).rename(
        columns={
            "W-L%": "prev_win_pct",
            "SRS": "prev_srs",
            "SOS": "prev_sos",
        }
    )

    adv = _load_table(
        SEASONS_DIR / f"adv_{year}.csv",
        ["Pace", "ORtg", "FTr", "3PAr", "eFG%", "TOV%", "ORB%"],
    )
    adv["prev_off_ppp"] = adv["ORtg"] / 100.0
    adv["prev_pace"] = adv["Pace"]
    adv["prev_efg"] = adv["eFG%"]
    adv["prev_tov_rate"] = adv["TOV%"] / 100.0
    adv["prev_orb_rate"] = adv["ORB%"] / 100.0
    adv["prev_ftr"] = adv["FTr"]
    adv["prev_threepar"] = adv["3PAr"]
    adv = adv[
        [
            "team",
            "prev_off_ppp",
            "prev_pace",
            "prev_efg",
            "prev_tov_rate",
            "prev_orb_rate",
            "prev_ftr",
            "prev_threepar",
        ]
    ].copy()

    basic_shooting = _load_table(
        SEASONS_DIR / f"basic_{year}.csv",
        ["3P%", "FT%"],
    ).rename(
        columns={
            "3P%": "prev_three_pct",
            "FT%": "prev_ft_pct",
        }
    )
    basic_shooting = basic_shooting[
        [
            "team",
            "prev_three_pct",
            "prev_ft_pct",
        ]
    ].copy()

    opp = _load_table(
        SEASONS_DIR / f"opp_{year}.csv",
        ["3P%", "FT%"],
    ).rename(
        columns={
            "3P%": "prev_opp_three_pct",
            "FT%": "prev_opp_ft_pct",
        }
    )
    opp = opp[
        [
            "team",
            "prev_opp_three_pct",
            "prev_opp_ft_pct",
        ]
    ].copy()

    opp_adv = _load_table(
        SEASONS_DIR / f"opp_adv_{year}.csv",
        ["ORtg", "FTr", "3PAr", "eFG%", "TOV%", "ORB%"],
    )
    opp_adv["prev_def_ppp"] = opp_adv["ORtg"] / 100.0
    opp_adv["prev_opp_threepar"] = opp_adv["3PAr"]
    opp_adv["prev_opp_efg"] = opp_adv["eFG%"]
    opp_adv["prev_opp_tov_rate"] = opp_adv["TOV%"] / 100.0
    opp_adv["prev_opp_orb_rate"] = opp_adv["ORB%"] / 100.0
    opp_adv["prev_opp_ftr"] = opp_adv["FTr"]
    opp_adv = opp_adv[
        [
            "team",
            "prev_def_ppp",
            "prev_opp_threepar",
            "prev_opp_efg",
            "prev_opp_tov_rate",
            "prev_opp_orb_rate",
            "prev_opp_ftr",
        ]
    ].copy()

    season = (
        basic.merge(adv, on="team", how="outer")
        .merge(basic_shooting, on="team", how="outer")
        .merge(opp, on="team", how="outer")
        .merge(opp_adv, on="team", how="outer")
    )
    season["prev_margin_ppp"] = season["prev_off_ppp"] - season["prev_def_ppp"]
    season["season"] = year
    season = season.sort_values("team").reset_index(drop=True)
    return season


def build_all_season_features():
    years = sorted(int(path.stem.split("_")[-1]) for path in SEASONS_DIR.glob("basic_*.csv"))
    for year in years:
        season = build_season_features_for_year(year)
        season.to_csv(SEASONS_DIR / f"season_features_{year}.csv", index=False)


def load_season_features(year):
    path = SEASONS_DIR / f"season_features_{year}.csv"
    if not path.exists():
        build_all_season_features()

    return pd.read_csv(path)


if __name__ == "__main__":
    build_all_season_features()
