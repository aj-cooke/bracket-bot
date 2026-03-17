import argparse
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from bracket_workbook import (
    build_bracket_sheet,
    build_bracket_games,
    build_game_data_sheet,
    choose_winner,
    extract_pdf_layout_text,
    load_probability_lookup,
    parse_region_blocks,
    verify_workbook,
    write_workbook_atomic,
)
from tournament_matchups import default_field_source_url, fetch_bracket_pdf


DEFAULT_TOURNAMENT_DIR = Path("results/tournament")
DEFAULT_BRACKET_DIR = Path("results/bracket")
DEFAULT_OUTPUT_DIR = Path("results/bracket_optimized")
ROUND_WEIGHTS = {
    "First Four": 1,
    "Round of 64": 1,
    "Round of 32": 2,
    "Sweet 16": 4,
    "Elite 8": 8,
    "Final Four": 16,
    "Championship": 32,
}
EXPECTED_GAME_COUNT = 67
SCORE_EPSILON = 1e-12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season-year", type=int, default=2026)
    parser.add_argument("--cutoff-date", default="2026-03-15")
    parser.add_argument("--field-source-url")
    parser.add_argument("--tournament-dir")
    parser.add_argument("--probability-matrix-path")
    parser.add_argument("--team-field-path")
    parser.add_argument("--bracket-games-path")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def write_json_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(path)


def write_csv_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def build_team_meta(field_df):
    return {
        row.team_slug: {
            "team_slug": row.team_slug,
            "team_name": row.school,
            "seed": int(row.seed),
            "field_order": int(row.field_order),
        }
        for row in field_df.itertuples(index=False)
    }


def validate_probability_lookup(probability_lookup, team_meta):
    matrix_teams = set(probability_lookup.index)
    expected_teams = set(team_meta)
    missing = sorted(expected_teams - matrix_teams)
    if missing:
        raise ValueError(f"Probability matrix is missing tournament teams: {missing}")

    for team in expected_teams:
        row_teams = set(probability_lookup.columns)
        if team not in row_teams:
            raise ValueError(f"Probability matrix is missing column for team '{team}'")
        diagonal = float(probability_lookup.loc[team, team])
        if abs(diagonal - 0.5) > 1e-9:
            raise ValueError(f"Probability matrix diagonal must be 0.5 for team '{team}', found {diagonal}")

    max_error = 0.0
    ordered_teams = sorted(expected_teams)
    for team_a in ordered_teams:
        for team_b in ordered_teams:
            if team_a == team_b:
                continue
            error = abs(float(probability_lookup.loc[team_a, team_b]) + float(probability_lookup.loc[team_b, team_a]) - 1.0)
            max_error = max(max_error, error)
    if max_error > 1e-9:
        raise ValueError(f"Probability matrix symmetry check failed; max error {max_error}")


def load_games_from_cached_csv(path):
    df = pd.read_csv(path)
    required_cols = [
        "game_id",
        "round_order",
        "round_name",
        "region",
        "slot_label",
        "source_top_type",
        "source_top_id",
        "source_bottom_type",
        "source_bottom_id",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Cached bracket games file is missing required columns: {missing}")

    games = df[required_cols].drop_duplicates().sort_values(["round_order", "game_id"]).to_dict(orient="records")
    if len(games) != EXPECTED_GAME_COUNT:
        raise ValueError(f"Expected {EXPECTED_GAME_COUNT} cached games, found {len(games)}")
    return games


def load_bracket_games(args, field_df, default_bracket_games_path):
    cached_path = Path(args.bracket_games_path) if args.bracket_games_path else default_bracket_games_path
    if cached_path.exists():
        return load_games_from_cached_csv(cached_path), str(cached_path), "cached_bracket_games"

    field_source_url = args.field_source_url or default_field_source_url(args.season_year)
    pdf_bytes, resolved_url = fetch_bracket_pdf(field_source_url)
    pdf_text = extract_pdf_layout_text(pdf_bytes)
    region_blocks = parse_region_blocks(pdf_text)
    games = build_bracket_games(region_blocks, field_df)
    if len(games) != EXPECTED_GAME_COUNT:
        raise ValueError(f"Expected {EXPECTED_GAME_COUNT} parsed games, found {len(games)}")
    return games, resolved_url, "ncaa_pdf"


def sort_games(games):
    return sorted(games, key=lambda item: (int(item["round_order"]), item["game_id"]))


def make_source_ref(source_type, source_id):
    return source_type, source_id


def build_game_lookup(games):
    game_by_id = {game["game_id"]: game for game in games}
    if len(game_by_id) != len(games):
        raise ValueError("Duplicate game ids found in game graph")
    if "championship" not in game_by_id:
        raise ValueError("Missing championship game in game graph")
    return game_by_id


def team_sort_key(team_slug, team_meta):
    team = team_meta[team_slug]
    return (int(team["seed"]), int(team["field_order"]), team["team_slug"])


def choose_better_team(candidate_team, candidate_score, incumbent_team, incumbent_score, team_meta):
    if incumbent_team is None:
        return True
    if candidate_score > incumbent_score + SCORE_EPSILON:
        return True
    if abs(candidate_score - incumbent_score) <= SCORE_EPSILON:
        return team_sort_key(candidate_team, team_meta) < team_sort_key(incumbent_team, team_meta)
    return False


def compute_game_win_probabilities(games, probability_lookup):
    game_by_id = build_game_lookup(games)

    @lru_cache(maxsize=None)
    def reachable_teams(source_type, source_id):
        if source_type == "team":
            return (source_id,)
        game = game_by_id[source_id]
        top = reachable_teams(game["source_top_type"], game["source_top_id"])
        bottom = reachable_teams(game["source_bottom_type"], game["source_bottom_id"])
        return tuple(sorted(set(top) | set(bottom)))

    @lru_cache(maxsize=None)
    def source_distribution(source_type, source_id):
        if source_type == "team":
            return ((source_id, 1.0),)

        game = game_by_id[source_id]
        top_distribution = dict(source_distribution(game["source_top_type"], game["source_top_id"]))
        bottom_distribution = dict(source_distribution(game["source_bottom_type"], game["source_bottom_id"]))

        winner_probabilities = {}
        for top_team, top_prob in top_distribution.items():
            matchup_prob = 0.0
            for bottom_team, bottom_prob in bottom_distribution.items():
                matchup_prob += bottom_prob * float(probability_lookup.loc[top_team, bottom_team])
            winner_probabilities[top_team] = top_prob * matchup_prob

        for bottom_team, bottom_prob in bottom_distribution.items():
            matchup_prob = 0.0
            for top_team, top_prob in top_distribution.items():
                matchup_prob += top_prob * float(probability_lookup.loc[bottom_team, top_team])
            winner_probabilities[bottom_team] = bottom_prob * matchup_prob

        total_probability = sum(winner_probabilities.values())
        if abs(total_probability - 1.0) > 1e-9:
            raise ValueError(
                f"Winner probabilities for game '{source_id}' sum to {total_probability} instead of 1.0"
            )

        ordered = sorted(winner_probabilities.items())
        return tuple(ordered)

    game_win_probabilities = {
        game["game_id"]: dict(source_distribution("game", game["game_id"]))
        for game in sort_games(games)
    }
    reachable_lookup = {
        make_source_ref("game", game_id): set(reachable_teams("game", game_id))
        for game_id in game_by_id
    }
    return game_win_probabilities, reachable_lookup


def optimize_expected_bracket(games, team_meta, probability_lookup, round_weights=None):
    round_weights = round_weights or ROUND_WEIGHTS
    game_by_id = build_game_lookup(games)
    game_win_probabilities, reachable_lookup = compute_game_win_probabilities(games, probability_lookup)

    @lru_cache(maxsize=None)
    def reachable_teams(source_type, source_id):
        if source_type == "team":
            return frozenset([source_id])
        return frozenset(reachable_lookup[make_source_ref(source_type, source_id)])

    @lru_cache(maxsize=None)
    def optimize_source(source_type, source_id):
        if source_type == "team":
            return {
                "best_score_by_team": {source_id: 0.0},
                "best_any_team": source_id,
                "best_any_score": 0.0,
                "choice_by_team": {},
            }
        return optimize_game(source_id)

    @lru_cache(maxsize=None)
    def optimize_game(game_id):
        game = game_by_id[game_id]
        round_name = game["round_name"]
        round_points = round_weights[round_name]
        top_type = game["source_top_type"]
        top_id = game["source_top_id"]
        bottom_type = game["source_bottom_type"]
        bottom_id = game["source_bottom_id"]

        top_state = optimize_source(top_type, top_id)
        bottom_state = optimize_source(bottom_type, bottom_id)
        top_teams = reachable_teams(top_type, top_id)
        bottom_teams = reachable_teams(bottom_type, bottom_id)
        winner_probabilities = game_win_probabilities[game_id]

        best_score_by_team = {}
        choice_by_team = {}
        best_any_team = None
        best_any_score = None

        for team_slug in sorted(top_teams | bottom_teams):
            node_points = round_points * winner_probabilities.get(team_slug, 0.0)
            if team_slug in top_teams:
                top_score = top_state["best_score_by_team"][team_slug]
                bottom_team = bottom_state["best_any_team"]
                bottom_score = bottom_state["best_any_score"]
                total_score = node_points + top_score + bottom_score
                choice_by_team[team_slug] = {
                    "source_top_winner": team_slug,
                    "source_bottom_winner": bottom_team,
                }
            else:
                top_team = top_state["best_any_team"]
                top_score = top_state["best_any_score"]
                bottom_score = bottom_state["best_score_by_team"][team_slug]
                total_score = node_points + top_score + bottom_score
                choice_by_team[team_slug] = {
                    "source_top_winner": top_team,
                    "source_bottom_winner": team_slug,
                }

            best_score_by_team[team_slug] = total_score
            if choose_better_team(team_slug, total_score, best_any_team, best_any_score, team_meta):
                best_any_team = team_slug
                best_any_score = total_score

        return {
            "best_score_by_team": best_score_by_team,
            "best_any_team": best_any_team,
            "best_any_score": best_any_score,
            "choice_by_team": choice_by_team,
        }

    optimized_states = {game["game_id"]: optimize_game(game["game_id"]) for game in sort_games(games)}
    return {
        "game_win_probabilities": game_win_probabilities,
        "optimized_states": optimized_states,
        "champion": optimized_states["championship"]["best_any_team"],
        "expected_total_score": optimized_states["championship"]["best_any_score"],
        "round_weights": round_weights,
    }


def reconstruct_optimized_picks(games, optimized_result):
    game_by_id = build_game_lookup(games)
    optimized_states = optimized_result["optimized_states"]
    picks = {}

    def resolve_source(source_type, source_id, forced_team=None):
        if source_type == "team":
            return source_id

        state = optimized_states[source_id]
        chosen_winner = forced_team if forced_team is not None else state["best_any_team"]
        choice = state["choice_by_team"][chosen_winner]
        game = game_by_id[source_id]
        top_team = resolve_source(game["source_top_type"], game["source_top_id"], choice["source_top_winner"])
        bottom_team = resolve_source(
            game["source_bottom_type"],
            game["source_bottom_id"],
            choice["source_bottom_winner"],
        )
        picks[source_id] = {
            "team_a_slug": top_team,
            "team_b_slug": bottom_team,
            "picked_winner_slug": chosen_winner,
        }
        return chosen_winner

    resolve_source("game", "championship")
    return picks


def build_greedy_picks(games, team_meta, probability_lookup):
    game_by_id = build_game_lookup(games)
    picks = {}

    def resolve_source(source_type, source_id):
        if source_type == "team":
            return source_id

        if source_id in picks:
            return picks[source_id]["picked_winner_slug"]

        game = game_by_id[source_id]
        team_a_slug = resolve_source(game["source_top_type"], game["source_top_id"])
        team_b_slug = resolve_source(game["source_bottom_type"], game["source_bottom_id"])
        prob_a = float(probability_lookup.loc[team_a_slug, team_b_slug])
        winner_slug, _ = choose_winner(team_meta[team_a_slug], team_meta[team_b_slug], prob_a)
        picks[source_id] = {
            "team_a_slug": team_a_slug,
            "team_b_slug": team_b_slug,
            "picked_winner_slug": winner_slug,
        }
        return winner_slug

    for game in sort_games(games):
        resolve_source("game", game["game_id"])

    return picks


def build_game_rows(games, picks, team_meta, probability_lookup, game_win_probabilities, round_weights):
    rows = []
    total_expected_points = 0.0

    for game in sort_games(games):
        game_id = game["game_id"]
        team_a_slug = picks[game_id]["team_a_slug"]
        team_b_slug = picks[game_id]["team_b_slug"]
        picked_winner_slug = picks[game_id]["picked_winner_slug"]
        team_a = team_meta[team_a_slug]
        team_b = team_meta[team_b_slug]
        picked_winner = team_meta[picked_winner_slug]
        picked_loser = team_b if picked_winner_slug == team_a_slug else team_a
        prob_a = float(probability_lookup.loc[team_a_slug, team_b_slug])
        prob_b = 1.0 - prob_a
        direct_pick_prob = prob_a if picked_winner_slug == team_a_slug else prob_b
        node_win_prob = float(game_win_probabilities[game_id][picked_winner_slug])
        round_points = int(round_weights[game["round_name"]])
        expected_points = round_points * node_win_prob
        total_expected_points += expected_points

        rows.append(
            {
                **game,
                "team_a_slug": team_a_slug,
                "team_a_name": team_a["team_name"],
                "seed_a": int(team_a["seed"]),
                "team_b_slug": team_b_slug,
                "team_b_name": team_b["team_name"],
                "seed_b": int(team_b["seed"]),
                "win_prob_a": prob_a,
                "win_prob_b": prob_b,
                "picked_winner_slug": picked_winner_slug,
                "picked_winner_name": picked_winner["team_name"],
                "picked_seed": int(picked_winner["seed"]),
                "picked_winner_prob": direct_pick_prob,
                "picked_loser_slug": picked_loser["team_slug"],
                "picked_loser_name": picked_loser["team_name"],
                "picked_loser_seed": int(picked_loser["seed"]),
                "round_points": round_points,
                "picked_team_game_win_prob": node_win_prob,
                "expected_points": expected_points,
            }
        )

    if len(rows) != len(games):
        raise ValueError(f"Expected {len(games)} output rows, found {len(rows)}")

    return pd.DataFrame(rows), total_expected_points


def main():
    args = parse_args()
    tournament_dir = Path(args.tournament_dir) if args.tournament_dir else DEFAULT_TOURNAMENT_DIR / str(args.season_year)
    probability_matrix_path = (
        Path(args.probability_matrix_path) if args.probability_matrix_path else tournament_dir / "probability_matrix.csv"
    )
    team_field_path = Path(args.team_field_path) if args.team_field_path else tournament_dir / "team_field.csv"
    default_bracket_games_path = DEFAULT_BRACKET_DIR / str(args.season_year) / "bracket_games.csv"
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / str(args.season_year)
    workbook_out_path = output_dir / "bracket.xlsx"
    bracket_games_out_path = output_dir / "bracket_games.csv"
    run_meta_path = output_dir / "run_meta.json"

    field_df = pd.read_csv(team_field_path)
    team_meta = build_team_meta(field_df)
    probability_lookup = load_probability_lookup(probability_matrix_path)
    validate_probability_lookup(probability_lookup, team_meta)

    games, bracket_source_reference, bracket_source_kind = load_bracket_games(args, field_df, default_bracket_games_path)

    optimized_result = optimize_expected_bracket(games, team_meta, probability_lookup, ROUND_WEIGHTS)
    optimized_picks = reconstruct_optimized_picks(games, optimized_result)
    optimized_rows, optimized_total = build_game_rows(
        games,
        optimized_picks,
        team_meta,
        probability_lookup,
        optimized_result["game_win_probabilities"],
        ROUND_WEIGHTS,
    )

    greedy_picks = build_greedy_picks(games, team_meta, probability_lookup)
    greedy_rows, greedy_total = build_game_rows(
        games,
        greedy_picks,
        team_meta,
        probability_lookup,
        optimized_result["game_win_probabilities"],
        ROUND_WEIGHTS,
    )

    champion_name = team_meta[optimized_result["champion"]]["team_name"]
    optimized_game_results = {
        row["game_id"]: row
        for row in optimized_rows.to_dict(orient="records")
    }
    workbook = Workbook()
    build_bracket_sheet(
        workbook,
        optimized_game_results,
        champion_name,
        subtitle="Picked to maximize expected bracket score from the model probability matrix",
    )
    build_game_data_sheet(workbook, optimized_rows)
    write_workbook_atomic(workbook, workbook_out_path)
    verify_workbook(workbook_out_path)

    write_csv_atomic(optimized_rows, bracket_games_out_path)
    write_json_atomic(
        {
            "season_year": args.season_year,
            "cutoff_date": args.cutoff_date,
            "team_field_path": str(team_field_path),
            "probability_matrix_path": str(probability_matrix_path),
            "bracket_source_kind": bracket_source_kind,
            "bracket_source_reference": bracket_source_reference,
            "output_path": str(workbook_out_path),
            "bracket_games_path": str(bracket_games_out_path),
            "game_rows": int(len(optimized_rows)),
            "round_weights": ROUND_WEIGHTS,
            "champion": champion_name,
            "champion_slug": optimized_result["champion"],
            "expected_total_score": optimized_total,
            "greedy_expected_total_score": greedy_total,
            "expected_score_gain_vs_greedy": optimized_total - greedy_total,
            "changed_pick_count_vs_greedy": int(
                (optimized_rows["picked_winner_slug"] != greedy_rows["picked_winner_slug"]).sum()
            ),
        },
        run_meta_path,
    )

    print(f"Saved workbook to {workbook_out_path}")
    print(f"Saved optimized bracket games to {bracket_games_out_path}")
    print(f"Champion: {champion_name}")
    print(f"Expected score: {optimized_total:.6f}")
    print(f"Greedy expected score: {greedy_total:.6f}")


if __name__ == "__main__":
    main()
