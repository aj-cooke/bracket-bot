import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

from season_priors import _normalize_school_name, _slugify_school
from training import prepare_features
from train_test_split import DIRECT_PRIOR_COLS, FACTOR_COLS


DEFAULT_OUTPUT_DIR = Path("results/tournament")
DEFAULT_MODEL_PATH = Path("models/production/xgb_model.json")
DEFAULT_MODEL_META_PATH = Path("models/production/xgb_model_meta.json")
YEAR_FILE_TEMPLATE = "all_games_{year}_with_sos_features.csv"
BRACKET_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
RECORD_AFTER_PATTERN = re.compile(
    r"^(?P<seed>1[0-6]|[1-9])\s+(?P<team>.+?)\s+\((?P<wins>\d+)-(?P<losses>\d+)\)$"
)
RECORD_BEFORE_PATTERN = re.compile(
    r"^\((?P<wins>\d+)-(?P<losses>\d+)\)\s+(?P<team>.+?)\s+(?P<seed>1[0-6]|[1-9])$"
)
BRACKET_TEAM_ALIASES = {
    "byu": "brigham-young",
    "cal-baptist": "california-baptist",
    "iowa-st": "iowa-state",
    "kennesaw-st": "kennesaw-state",
    "long-island": "long-island-university",
    "mcneese": "mcneese-state",
    "miami-ohio": "miami-oh",
    "michigan-st": "michigan-state",
    "north-dakota-st": "north-dakota-state",
    "ohio-st": "ohio-state",
    "penn": "pennsylvania",
    "prairie-view-am": "prairie-view",
    "saint-marys": "saint-marys-ca",
    "smu": "southern-methodist",
    "st-johns": "st-johns-ny",
    "tennessee-st": "tennessee-state",
    "uconn": "connecticut",
    "umbc": "maryland-baltimore-county",
    "utah-st": "utah-state",
    "vcu": "virginia-commonwealth",
    "wright-st": "wright-state",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season-year", type=int, default=2026)
    parser.add_argument("--cutoff-date", default="2026-03-15")
    parser.add_argument("--field-source-url")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-meta-path", default=str(DEFAULT_MODEL_META_PATH))
    return parser.parse_args()


def default_field_source_url(season_year):
    return f"https://www.ncaa.com/brackets/print/basketball-men/d1/{season_year}?%24web_only=true"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


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


def require_pdftotext():
    if shutil.which("pdftotext") is None:
        raise RuntimeError("Missing required dependency: pdftotext")


def fetch_bracket_pdf(url):
    response = requests.get(url, timeout=30, headers={"User-Agent": BRACKET_USER_AGENT})
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise ValueError(f"Bracket source did not return a PDF: {content_type or 'unknown content type'}")
    return response.content, response.url


def extract_pdf_text(pdf_bytes):
    require_pdftotext()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
        pdf_file.write(pdf_bytes)
        pdf_file.flush()
        return subprocess.check_output(["pdftotext", pdf_file.name, "-"]).decode("utf-8", errors="ignore")


def parse_bracket_field(pdf_text):
    rows = []
    for raw_line in pdf_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = RECORD_AFTER_PATTERN.match(line) or RECORD_BEFORE_PATTERN.match(line)
        if not match:
            continue
        rows.append(
            {
                "seed": int(match.group("seed")),
                "bracket_team_name": match.group("team"),
                "wins": int(match.group("wins")),
                "losses": int(match.group("losses")),
            }
        )

    field_df = pd.DataFrame(rows).drop_duplicates(subset=["bracket_team_name"]).reset_index(drop=True)
    if len(field_df) != 68:
        raise ValueError(f"Expected 68 unique tournament teams, found {len(field_df)}")
    field_df["field_order"] = np.arange(1, len(field_df) + 1)
    return field_df


def load_team_snapshots(season_year, cutoff_date):
    path = Path("data/years") / YEAR_FILE_TEMPLATE.format(year=season_year)
    if not path.exists():
        raise FileNotFoundError(f"Missing season feature file: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff_ts = pd.Timestamp(cutoff_date)
    df = df[df["Date"] <= cutoff_ts].copy()
    if df.empty:
        raise ValueError(f"No season rows found on or before cutoff date {cutoff_date}")

    snapshots = df.sort_values(["team", "Date", "Gtm"]).groupby("team", as_index=False).tail(1).copy()
    snapshots["Date"] = snapshots["Date"].dt.strftime("%Y-%m-%d")
    return snapshots


def map_field_to_local_teams(field_df, snapshots):
    snapshots = snapshots.copy()
    available_teams = set(snapshots["team"])
    school_map = {}
    for row in snapshots[["team", "School"]].drop_duplicates().itertuples(index=False):
        normalized_school = _normalize_school_name(row.School)
        school_map[normalized_school] = row.team

    mapped_rows = []
    unmapped_rows = []
    for row in field_df.itertuples(index=False):
        normalized_name = _normalize_school_name(row.bracket_team_name)
        slug_candidate = _slugify_school(normalized_name)
        slug_candidate = BRACKET_TEAM_ALIASES.get(slug_candidate, slug_candidate)

        resolved_team = None
        if slug_candidate in available_teams:
            resolved_team = slug_candidate
        elif normalized_name in school_map:
            resolved_team = school_map[normalized_name]

        if resolved_team is None:
            unmapped_rows.append(
                {
                    "bracket_team_name": row.bracket_team_name,
                    "normalized_team_name": normalized_name,
                    "slug_candidate": slug_candidate,
                }
            )
            continue

        snapshot_row = snapshots[snapshots["team"] == resolved_team].iloc[0]
        mapped_rows.append(
            {
                "field_order": row.field_order,
                "seed": row.seed,
                "wins": row.wins,
                "losses": row.losses,
                "bracket_team_name": row.bracket_team_name,
                "normalized_team_name": normalized_name,
                "team_slug": resolved_team,
                "school": snapshot_row["School"],
                "snapshot_date": snapshot_row["Date"],
                "snapshot_gtm": int(snapshot_row["Gtm"]),
            }
        )

    if unmapped_rows:
        unmapped_df = pd.DataFrame(unmapped_rows)
        raise ValueError("Failed to map tournament teams to local slugs:\n" + unmapped_df.to_string(index=False))

    mapped_df = pd.DataFrame(mapped_rows)
    if mapped_df["team_slug"].duplicated().any():
        dupes = mapped_df[mapped_df["team_slug"].duplicated(keep=False)].sort_values("team_slug")
        raise ValueError("Tournament team mapping produced duplicate local slugs:\n" + dupes.to_string(index=False))

    if len(mapped_df) != 68:
        raise ValueError(f"Expected 68 mapped tournament teams, found {len(mapped_df)}")

    return mapped_df.sort_values("field_order").reset_index(drop=True)


def build_snapshot_matchup_features(team_snapshots):
    snapshots = team_snapshots.copy()
    snapshots["_cross_key"] = 1
    matchup_df = snapshots.merge(snapshots, on="_cross_key", suffixes=("_team_a", "_team_b")).drop(columns="_cross_key")
    matchup_df = matchup_df[matchup_df["team_team_a"] != matchup_df["team_team_b"]].copy()
    derived = {
        "site_team_a": pd.Series("N", index=matchup_df.index, dtype="object"),
        "Type_team_a": pd.Series("CTOURN", index=matchup_df.index, dtype="object"),
    }

    state_cols = [
        col
        for col in team_snapshots.columns
        if col.startswith("blend_ewm_fast_")
        or col.startswith("blend_ewm_slow_")
        or col.startswith("blend_reg_adj_")
    ]

    for col in state_cols:
        a_col = f"{col}_team_a"
        b_col = f"{col}_team_b"
        derived[f"matchup_delta_{col}"] = matchup_df[a_col] - matchup_df[b_col]
        if col.endswith("pace"):
            derived[f"matchup_mean_{col}"] = (matchup_df[a_col] + matchup_df[b_col]) / 2.0

    for col in DIRECT_PRIOR_COLS:
        derived[f"matchup_delta_{col}"] = matchup_df[f"{col}_team_a"] - matchup_df[f"{col}_team_b"]

    derived["matchup_mean_blend_ewm_fast_threepar"] = (
        matchup_df["blend_ewm_fast_threepar_team_a"] + matchup_df["blend_ewm_fast_threepar_team_b"]
    ) / 2.0
    derived["matchup_mean_blend_ewm_slow_threepar"] = (
        matchup_df["blend_ewm_slow_threepar_team_a"] + matchup_df["blend_ewm_slow_threepar_team_b"]
    ) / 2.0
    derived["matchup_3par_edge_fast"] = (
        matchup_df["blend_ewm_fast_threepar_team_a"] - matchup_df["blend_ewm_fast_opp_threepar_team_b"]
    )
    derived["matchup_3par_edge_slow"] = (
        matchup_df["blend_ewm_slow_threepar_team_a"] - matchup_df["blend_ewm_slow_opp_threepar_team_b"]
    )
    derived["matchup_opp_3par_edge_fast"] = (
        matchup_df["blend_ewm_fast_opp_threepar_team_a"] - matchup_df["blend_ewm_fast_threepar_team_b"]
    )
    derived["matchup_opp_3par_edge_slow"] = (
        matchup_df["blend_ewm_slow_opp_threepar_team_a"] - matchup_df["blend_ewm_slow_threepar_team_b"]
    )

    derived["matchup_off_edge_fast"] = (
        matchup_df["blend_ewm_fast_off_ppp_team_a"] - matchup_df["blend_ewm_fast_def_ppp_team_b"]
    )
    derived["matchup_off_edge_slow"] = (
        matchup_df["blend_ewm_slow_off_ppp_team_a"] - matchup_df["blend_ewm_slow_def_ppp_team_b"]
    )
    derived["matchup_def_edge_fast"] = (
        matchup_df["blend_ewm_fast_def_ppp_team_a"] - matchup_df["blend_ewm_fast_off_ppp_team_b"]
    )
    derived["matchup_def_edge_slow"] = (
        matchup_df["blend_ewm_slow_def_ppp_team_a"] - matchup_df["blend_ewm_slow_off_ppp_team_b"]
    )
    derived["matchup_adj_edge_fast"] = (
        matchup_df["blend_reg_adj_fast_margin_ppp_team_a"] - matchup_df["blend_reg_adj_fast_margin_ppp_team_b"]
    )
    derived["matchup_adj_edge_slow"] = (
        matchup_df["blend_reg_adj_slow_margin_ppp_team_a"] - matchup_df["blend_reg_adj_slow_margin_ppp_team_b"]
    )
    derived["matchup_adj_off_edge_fast"] = (
        matchup_df["blend_reg_adj_fast_off_ppp_team_a"] - matchup_df["blend_reg_adj_fast_def_ppp_team_b"]
    )
    derived["matchup_adj_off_edge_slow"] = (
        matchup_df["blend_reg_adj_slow_off_ppp_team_a"] - matchup_df["blend_reg_adj_slow_def_ppp_team_b"]
    )
    derived["matchup_adj_def_edge_fast"] = (
        matchup_df["blend_reg_adj_fast_def_ppp_team_a"] - matchup_df["blend_reg_adj_fast_off_ppp_team_b"]
    )
    derived["matchup_adj_def_edge_slow"] = (
        matchup_df["blend_reg_adj_slow_def_ppp_team_a"] - matchup_df["blend_reg_adj_slow_off_ppp_team_b"]
    )

    for horizon in ["fast", "slow"]:
        for factor in FACTOR_COLS:
            derived[f"matchup_adj_{factor}_edge_{horizon}"] = (
                matchup_df[f"blend_reg_adj_{horizon}_{factor}_team_a"]
                - matchup_df[f"blend_reg_adj_{horizon}_opp_{factor}_team_b"]
            )
            derived[f"matchup_adj_opp_{factor}_edge_{horizon}"] = (
                matchup_df[f"blend_reg_adj_{horizon}_opp_{factor}_team_a"]
                - matchup_df[f"blend_reg_adj_{horizon}_{factor}_team_b"]
            )

    base_df = matchup_df.drop(columns=["site_team_a", "Type_team_a"])
    derived_df = pd.DataFrame(derived, index=matchup_df.index)
    return pd.concat([base_df, derived_df], axis=1)


def prepare_inference_features(matchup_df, expected_columns):
    X = matchup_df[expected_columns].copy()
    actual_columns = list(X.columns)
    if actual_columns != expected_columns:
        raise ValueError("Inference feature columns do not match production model feature columns")
    X, _ = prepare_features(X)
    return X


def load_model(model_path):
    model = XGBClassifier(enable_categorical=True)
    model.load_model(model_path)
    return model


def build_directional_output(matchup_df, field_df, probabilities):
    field_lookup = field_df.set_index("team_slug")[["school", "seed", "field_order"]]
    directional = matchup_df[
        [
            "team_team_a",
            "team_team_b",
        ]
    ].copy()
    directional = directional.rename(columns={"team_team_a": "team_a", "team_team_b": "team_b"})
    directional["raw_prob_team_a_beats_team_b"] = probabilities
    directional["team_a_name"] = directional["team_a"].map(field_lookup["school"])
    directional["team_b_name"] = directional["team_b"].map(field_lookup["school"])
    directional["seed_a"] = directional["team_a"].map(field_lookup["seed"])
    directional["seed_b"] = directional["team_b"].map(field_lookup["seed"])
    directional["field_order_a"] = directional["team_a"].map(field_lookup["field_order"])
    directional["field_order_b"] = directional["team_b"].map(field_lookup["field_order"])
    return directional


def build_averaged_output(directional, field_df):
    rows = []
    directional_map = directional.set_index(["team_a", "team_b"])["raw_prob_team_a_beats_team_b"].to_dict()
    field_lookup = field_df.set_index("team_slug")[["school", "seed", "field_order"]]
    ordered_teams = field_df["team_slug"].tolist()

    for i, team_i in enumerate(ordered_teams):
        for team_j in ordered_teams[i + 1 :]:
            prob_ij = directional_map[(team_i, team_j)]
            prob_ji = directional_map[(team_j, team_i)]
            avg_prob = (prob_ij + (1.0 - prob_ji)) / 2.0
            rows.append(
                {
                    "team_i": team_i,
                    "team_j": team_j,
                    "team_i_name": field_lookup.loc[team_i, "school"],
                    "team_j_name": field_lookup.loc[team_j, "school"],
                    "seed_i": int(field_lookup.loc[team_i, "seed"]),
                    "seed_j": int(field_lookup.loc[team_j, "seed"]),
                    "field_order_i": int(field_lookup.loc[team_i, "field_order"]),
                    "field_order_j": int(field_lookup.loc[team_j, "field_order"]),
                    "raw_prob_team_i_beats_team_j": prob_ij,
                    "raw_prob_team_j_beats_team_i": prob_ji,
                    "avg_prob_team_i_beats_team_j": avg_prob,
                    "avg_prob_team_j_beats_team_i": 1.0 - avg_prob,
                }
            )

    averaged = pd.DataFrame(rows)
    expected_rows = len(ordered_teams) * (len(ordered_teams) - 1) // 2
    if len(averaged) != expected_rows:
        raise ValueError(f"Expected {expected_rows} averaged matchup rows, found {len(averaged)}")
    return averaged


def build_probability_matrix(averaged, field_df):
    ordered_teams = field_df["team_slug"].tolist()
    matrix = pd.DataFrame(0.5, index=ordered_teams, columns=ordered_teams, dtype=float)
    for row in averaged.itertuples(index=False):
        matrix.loc[row.team_i, row.team_j] = row.avg_prob_team_i_beats_team_j
        matrix.loc[row.team_j, row.team_i] = row.avg_prob_team_j_beats_team_i

    matrix.index.name = "team_slug"
    return matrix.reset_index()


def save_outputs(output_dir, field_df, directional, averaged, matrix, metadata):
    write_csv_atomic(field_df, output_dir / "team_field.csv")
    write_csv_atomic(directional, output_dir / "matchups_directional.csv")
    write_csv_atomic(averaged, output_dir / "matchups_averaged.csv")
    write_csv_atomic(matrix, output_dir / "probability_matrix.csv")
    write_json_atomic(metadata, output_dir / "run_meta.json")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / str(args.season_year)
    field_source_url = args.field_source_url or default_field_source_url(args.season_year)
    model_path = Path(args.model_path)
    model_meta_path = Path(args.model_meta_path)

    model_meta = load_json(model_meta_path)
    expected_columns = model_meta["feature_columns"]

    pdf_bytes, resolved_url = fetch_bracket_pdf(field_source_url)
    pdf_text = extract_pdf_text(pdf_bytes)
    field_raw = parse_bracket_field(pdf_text)

    snapshots = load_team_snapshots(args.season_year, args.cutoff_date)
    field_df = map_field_to_local_teams(field_raw, snapshots)
    tournament_snapshots = snapshots[snapshots["team"].isin(field_df["team_slug"])].copy()
    if tournament_snapshots["team"].nunique() != len(field_df):
        raise ValueError("Tournament snapshot selection did not produce one row per tournament team")

    matchup_df = build_snapshot_matchup_features(tournament_snapshots)
    X_matchups = prepare_inference_features(matchup_df, expected_columns)
    model = load_model(model_path)
    probabilities = model.predict_proba(X_matchups)[:, 1]

    directional = build_directional_output(matchup_df, field_df, probabilities)
    averaged = build_averaged_output(directional, field_df)
    matrix = build_probability_matrix(averaged, field_df)

    metadata = {
        "season_year": args.season_year,
        "cutoff_date": args.cutoff_date,
        "field_source_url": field_source_url,
        "resolved_field_source_url": resolved_url,
        "model_path": str(model_path),
        "model_meta_path": str(model_meta_path),
        "model_feature_count": len(expected_columns),
        "team_count": int(len(field_df)),
        "directional_rows": int(len(directional)),
        "averaged_rows": int(len(averaged)),
    }
    save_outputs(output_dir, field_df, directional, averaged, matrix, metadata)

    print(f"Saved tournament outputs to {output_dir}")
    print(f"Teams: {len(field_df)}")
    print(f"Directional rows: {len(directional)}")
    print(f"Averaged rows: {len(averaged)}")


if __name__ == "__main__":
    main()
