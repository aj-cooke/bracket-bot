# bracket-bot

Prediction modeling for NCAA men's March Madness. The data pipeline scrapes season-level and game-level data from sport-reference.com, turns it into opponent-adjusted team state features, trains a matchup model, scores every tournament pairing, and then converts those probabilities into both a greedy bracket and an expected-value-optimized bracket according to the CBS bracket scoring system (1-2-4-8-16-32).

## Analysis Flow

The approach can be broken down into 5 stages:

1. Scrape and normalize historical college basketball data.
2. Build per-possession based features and rolling team state features (exclusive of prior games to score the next one).
3. Train a binary matchup model on historical games (e.g. whether or not team A will win.)
4. Freeze the tournament field at a pre-tournament cutoff date and score all pairwise matchups, defining the "greedy" bracket.
5. Run another set of picks via an optimization layer that maximizes expected bracket point total.

The main outputs are:

- season-level prior files in `data/seasons/`
- game-level feature files in `data/years/`
- training datasets in `data/train_test_files/`
- trained XGBoost models in `models/`
- tournament matchup probabilities in `results/tournament/<season>/`
- greedy and optimized bracket workbooks in `results/bracket/` and `results/bracket_optimized/`

## Data Scraping

The modeling stack starts from Sports Reference data.

- `scripts/scrape_seasons.py` pulls season summary tables for each year, including basic team stats, opponent stats, advanced team stats, and advanced opponent stats.
- Those season tables are used to create previous-season priors in `scripts/season_priors.py`.
- Game-level files under `data/years/` are then cleaned and standardized into a row-wise historical game dataset.

Once the raw data is saved, downstream scripts operate on the saved CSVs.

## Feature Creation

### Per-possession baseline

Raw box score results are converted into pace-adjusted features in `scripts/clean_games.py`. Instead of treating a 90-point game and a 65-point game as directly comparable, the pipeline estimates possessions and works in per-possession terms:

- offensive points per possession
- defensive points per possession
- per-possession margin
- pace

That keeps fast teams from looking artificially stronger just because they create more possessions.

### Practical strength-of-schedule adjustment

The adjusted layer in `scripts/opponent_features.py` uses weighted ridge regression snapshots to estimate team strength against schedule context. In practice, that means the model is not just looking at what a team did, but also at who it did it against, and who they did it against, and who they did it against ...

The important idea is recursive strength-of-schedule adjustment:

- offensive ratings are interpreted through the defenses faced
- defensive ratings are interpreted through the offenses faced
- tempo and four-factor style components are adjusted the same way

This is a practical regression-based approximation, not a full KenPom-style fixed-point solver. This has enabled much quicker iteration & feature experimentation though it could be a notable future performance improvement opportunity.

### Fast and slow smoothing

The model keeps two views of team form:

- fast recency weighting with a 14-day half-life
- slow recency weighting with a 45-day half-life

The fast view reacts more aggressively to late-season changes. The slow view is steadier and less sensitive to short streaks.

### Early-season priors

Early in a season, raw in-season samples are thin. To reduce noise, the pipeline blends in previous-season team summaries built in `scripts/season_priors.py`.

Those priors include broad team-quality and style indicators such as:

- previous offensive and defensive efficiency
- prior pace
- shooting and shot-profile measures
- opponent shooting and opponent shot-profile measures
- summary values like SRS, SOS, and win percentage

The prior weight decays as games accumulate. The currently locked schedule is:

- `prior_weight = exp(-prior_games / 4.0)`

That gives the model a reasonable starting point in November while still letting current-season evidence take over quickly.

### Matchup construction

After team-state features are built, the training set converts them into matchup features in `scripts/train_test_split.py`. The exact column set is large, but the high-level pattern is simple:

- team A offense vs team B defense
- team A defense vs team B offense
- fast-horizon and slow-horizon versions of the same idea
- strength-of-schedule adjusted efficiency and four-factor edges
- neutral-site / game-type context columns

## Model and Parameter Tuning

The game model is an XGBoost binary classifier trained to predict the probability that team A beats team B.

The training workflow in `scripts/training.py` intentionally separates two modes:

- `benchmark`: use a fixed, out of time holdout workflow for parameter tuning, selection, and validation
- `production`: retrain on all eligible pre-tournament data using the locked benchmark spec

The current retained benchmark spec is:

- `max_depth = 4`
- `learning_rate = 0.025`
- `n_estimators = 563`

On the locked benchmark holdout, that configuration produced:

- validation log loss: `0.5299`
- validation AUC: `0.8063`
- validation accuracy: `0.7221`

On the 2026 production retrain through `2026-03-15`, the in-sample training metrics were:

- training log loss: `0.5069`
- training AUC: `0.8269`
- training accuracy: `0.7408`

## Tournament Inference

Once a production model is trained, `scripts/tournament_matchups.py` freezes each tournament team at its latest pre-tournament state and scores every ordered pairing in the field. That stage produces a symmetric team-vs-team probability matrix. This probability matrix is the bridge from the regular-season model to actual bracket logic.

## Optimization Layer

The repo supports two bracket-generation approaches:

- a greedy bracket that picks the most likely winner of each game iteratively from the saved probability matrix
- an optimized bracket that maximizes expected total bracket points over the entire tournament tree

The optimization step lives in `scripts/optimize_bracket.py`. It uses round weights:

- `First Four = 1`
- `Round of 64 = 1`
- `Round of 32 = 2`
- `Sweet 16 = 4`
- `Elite 8 = 8`
- `Final Four = 16`
- `Championship = 32`

This is not a generic MILP or black-box solver. Because the bracket is a tree, the optimizer can solve the problem exactly with dynamic programming:

- compute each team's probability of surviving to and through each game
- evaluate expected points at every game node
- choose the coherent set of picks that maximizes total expected score, not just local win probability

That matters because the highest-probability pick in one game is not always the highest expected-value pick once downstream rounds are considered.

For the saved `2026-03-15` run, the expected bracket scores were:

- greedy bracket: `100.5446`
- optimized bracket: `100.8273`
- expected gain from optimization: `0.2827`
- changed picks vs greedy: `4`

### Rebuild upstream features

```bash
python3 scripts/clean_games.py
python3 scripts/opponent_features.py
```

### Build training datasets

Benchmark split:

```bash
python3 scripts/train_test_split.py --mode benchmark
```

Production split through a cutoff date:

```bash
python3 scripts/train_test_split.py --mode production --current-season-year 2026 --cutoff-date 2026-03-15
```

### Train models

Benchmark tuning sweep:

```bash
python3 scripts/training.py --mode benchmark
```

Production retrain with locked benchmark parameters:

```bash
python3 scripts/training.py --mode production
```

### Score the tournament field

```bash
python3 scripts/tournament_matchups.py --season-year 2026 --cutoff-date 2026-03-15
```

Outputs land in:

- `results/tournament/2026/team_field.csv`
- `results/tournament/2026/matchups_directional.csv`
- `results/tournament/2026/matchups_averaged.csv`
- `results/tournament/2026/probability_matrix.csv`
- `results/tournament/2026/run_meta.json`

### Generate the greedy bracket workbook

```bash
python3 -m pip install --user openpyxl
python3 scripts/bracket_workbook.py --season-year 2026 --cutoff-date 2026-03-15
```

### Generate the optimized bracket

```bash
python3 scripts/optimize_bracket.py --season-year 2026 --cutoff-date 2026-03-15
```

If the greedy bracket game graph already exists, reuse it:

```bash
python3 scripts/optimize_bracket.py \
  --season-year 2026 \
  --cutoff-date 2026-03-15 \
  --bracket-games-path results/bracket/2026/bracket_games.csv
```

## Conclusions
The optimization layer had only a few different picks from the greedy bracket, but they had nearly identical expected points. Rather than identifying opportunities from better downstream matchups that propagate a team through the tournament, the optimizer flipped very close matchups that faced a tough opponent in the subsequent round, hence the similar expected points. Accordingly, I actually ended up just using the greedy bracket since intuitively, it has lower variance. Interestingly, it also had more seed-wise upsets! I'll be sure to report back with results.
