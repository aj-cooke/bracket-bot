# bracket-bot
Prediction modeling for NCAA men's march madness

## Quick start (Sportradar NCAAMB)

1. Copy env template:

```bash
cp .env.example .env
```

2. Put your API key in `.env` as `SPORTRADAR_API_KEY=...`

3. List all seasons your API key can access:

```bash
python3 scripts/fetch_ncaamb.py list-seasons --out data/seasons.json
```

4. Fetch inferred current season:

```bash
python3 scripts/fetch_ncaamb.py current-season --out data/current_season.json
```

5. Fetch a season schedule (example: 2025 postseason):

```bash
python3 scripts/fetch_ncaamb.py season-schedule --year 2025 --season-type PST --out data/2025_pst_schedule.json
```

6. Fetch one game summary:

```bash
python3 scripts/fetch_ncaamb.py game-summary --game-id <game_uuid> --out data/game_summary.json
```

## Build a modeling dataset

Generate one CSV with game-level features (teams, score, basic box score stats) from season schedule + game summaries.

Single year:

```bash
python3 scripts/build_dataset.py --year 2025 --season-types REG,PST --out data/datasets/2025_reg_pst_games.csv
```

All accessible years for your key:

```bash
python3 scripts/build_dataset.py --all-seasons --season-types REG,CT,PST --out data/datasets/all_games.csv
```

Useful options:

- Bound years when pulling all: `--min-year 2012 --max-year 2025`
- Include not-yet-final games: `--include-incomplete`
- Increase/decrease API pacing: `--delay-seconds 0.2`

## Rate-limit tuning (important for trial keys)

If you hit `HTTP 429 Too Many Requests`, slow requests in `.env`:

```bash
SPORTRADAR_MIN_INTERVAL_SECONDS=1.5
SPORTRADAR_MAX_RETRIES=8
SPORTRADAR_BACKOFF_SECONDS=2.0
```

Then rerun the same command. The client now retries 429s automatically.

## Notes

- `SPORTRADAR_ACCESS_LEVEL` defaults to `trial`; switch to your paid access level if needed.
- Endpoints and defaults are configured in `src/sportradar_client.py`.
- If you shared your API key publicly, rotate it in Sportradar portal and replace it in `.env`.
