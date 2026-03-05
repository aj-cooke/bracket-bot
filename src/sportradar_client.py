import json
import os
import time
from datetime import date
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


class SportradarError(Exception):
    """Raised when the Sportradar API returns an error."""


@dataclass
class SportradarConfig:
    api_key: str
    package: str = "ncaamb"
    access_level: str = "trial"
    version: str = "v8"
    language: str = "en"
    base_domain: str = "api.sportradar.com"
    min_interval_seconds: float = 1.0
    max_retries: int = 6
    backoff_seconds: float = 1.5

    @classmethod
    def from_env(cls) -> "SportradarConfig":
        api_key = os.getenv("SPORTRADAR_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing SPORTRADAR_API_KEY. Put it in .env or your shell env.")

        return cls(
            api_key=api_key,
            package=os.getenv("SPORTRADAR_PACKAGE", "ncaamb"),
            access_level=os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial"),
            version=os.getenv("SPORTRADAR_VERSION", "v8"),
            language=os.getenv("SPORTRADAR_LANGUAGE", "en"),
            base_domain=os.getenv("SPORTRADAR_BASE_DOMAIN", "api.sportradar.com"),
            min_interval_seconds=float(os.getenv("SPORTRADAR_MIN_INTERVAL_SECONDS", "1.0")),
            max_retries=int(os.getenv("SPORTRADAR_MAX_RETRIES", "6")),
            backoff_seconds=float(os.getenv("SPORTRADAR_BACKOFF_SECONDS", "1.5")),
        )


class SportradarClient:
    def __init__(self, config: SportradarConfig):
        self.config = config
        self._last_request_at = 0.0

    def _respect_rate_limit(self) -> None:
        min_interval = max(0.0, self.config.min_interval_seconds)
        if min_interval <= 0:
            return
        elapsed = time.time() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    @staticmethod
    def _retry_after_seconds(exc: HTTPError) -> Optional[float]:
        retry_after = exc.headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except ValueError:
            return None

    def _build_url(self, path: str, extra_params: Optional[Dict[str, Any]] = None) -> str:
        path = path.lstrip("/")
        base = (
            f"https://{self.config.base_domain}/"
            f"{self.config.package}/{self.config.access_level}/{self.config.version}/"
            f"{self.config.language}/{path}"
        )

        params = {"api_key": self.config.api_key}
        if extra_params:
            params.update(extra_params)
        return f"{base}?{urlencode(params)}"

    def get(self, path: str, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._build_url(path, extra_params)
        attempts = 0
        while True:
            attempts += 1
            try:
                self._respect_rate_limit()
                with urlopen(url, timeout=30) as response:
                    self._last_request_at = time.time()
                    body = response.read().decode("utf-8")
                    return json.loads(body)
            except HTTPError as exc:
                self._last_request_at = time.time()
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempts <= self.config.max_retries:
                    retry_after = self._retry_after_seconds(exc)
                    if retry_after is None:
                        retry_after = self.config.backoff_seconds * (2 ** (attempts - 1))
                    wait_seconds = max(retry_after, 0.0)
                    print(
                        f"HTTP 429 for {path}; retry {attempts}/{self.config.max_retries} "
                        f"in {wait_seconds:.1f}s",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise SportradarError(f"HTTP {exc.code} for {url}: {detail}") from exc
            except URLError as exc:
                self._last_request_at = time.time()
                raise SportradarError(f"Network error for {url}: {exc.reason}") from exc

    def seasons(self) -> Dict[str, Any]:
        return self.get("league/seasons.json")

    def current_season(self) -> Dict[str, Any]:
        payload = self.seasons()
        seasons: List[Dict[str, Any]] = payload.get("seasons", []) or []
        if not seasons:
            raise SportradarError("No seasons returned by league/seasons endpoint")

        today = date.today()

        def season_date(value: Optional[str], fallback: date) -> date:
            if not value:
                return fallback
            try:
                return date.fromisoformat(value)
            except ValueError:
                return fallback

        def season_year(season: Dict[str, Any]) -> int:
            year_value = season.get("year")
            try:
                return int(year_value)
            except (TypeError, ValueError):
                return -1

        def recency_key(season: Dict[str, Any]) -> tuple:
            return (
                season_year(season),
                season_date(season.get("end_date"), date.min),
                season_date(season.get("start_date"), date.min),
            )

        # Prefer a season that currently spans today's date.
        in_range = [
            season
            for season in seasons
            if season_date(season.get("start_date"), date.min)
            <= today
            <= season_date(season.get("end_date"), date.max)
        ]
        if in_range:
            return max(in_range, key=recency_key)

        # Next, prefer actively in-progress status.
        in_progress = [s for s in seasons if (s.get("status") or "").lower() == "inprogress"]
        if in_progress:
            return max(in_progress, key=recency_key)

        # Fallback to the season with the latest end date.
        return max(seasons, key=recency_key)

    def season_schedule(self, year: int, season_type: str) -> Dict[str, Any]:
        # season_type examples: REG, PST, CT
        return self.get(f"games/{year}/{season_type}/schedule.json")

    def game_summary(self, game_id: str) -> Dict[str, Any]:
        return self.get(f"games/{game_id}/summary.json")
