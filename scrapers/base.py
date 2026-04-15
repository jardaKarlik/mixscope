"""
scrapers/base.py
================
Shared base class for all Mixscope scrapers.

Provides:
  - Standard constructor (config, gcs_client, bucket_name)
  - Rate-limited HTTP GET with exponential backoff on 429 / 5xx
  - GCS raw-backup helper
  - Abstract run() contract

All scrapers inherit from BaseScraper. Scrapers that need custom auth
(YouTube quota tracking, Spotify OAuth) override _get() but still call
super().__init__() and use _backup_to_gcs().
"""

import time
import json
import logging
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)


class BaseScraper:
    """
    Base class for all Mixscope scrapers.

    Subclass contract:
      - Set SOURCE = "mixcloud" / "youtube" / etc.  (used in GCS paths + logs)
      - Implement run() → dict with at minimum {"tracks": int, "errors": int}
    """

    SOURCE: str = ""

    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        self.config      = config
        self.gcs_client  = gcs_client
        self.bucket_name = bucket_name
        self._rps        = config.get("requests_per_second", 1)

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict = None,
             retries: int = 3, backoff: float = 2.0) -> dict:
        """
        Rate-limited GET with automatic retry/backoff.

        Behaviour:
          - Sleeps 1/rps before every attempt (simple token-bucket).
          - 429: honours Retry-After header; falls back to exponential backoff.
          - 5xx: retries up to `retries` times with exponential backoff.
          - 4xx (not 429): raises immediately — no retry.
          - Network errors: retries up to `retries` times.
        """
        base_delay = 1.0 / max(self._rps, 0.01)

        for attempt in range(retries + 1):
            time.sleep(base_delay)

            try:
                resp = requests.get(url, params=params or {}, timeout=15)
            except requests.RequestException as exc:
                if attempt < retries:
                    wait = backoff ** attempt
                    log.warning(
                        f"Network error (attempt {attempt + 1}/{retries + 1}): "
                        f"{exc} — retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                raise

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff ** (attempt + 1)))
                log.warning(
                    f"Rate limited (attempt {attempt + 1}/{retries + 1}) "
                    f"— waiting {wait:.0f}s"
                )
                time.sleep(wait)
                base_delay = 0  # already slept for this iteration
                continue

            if resp.status_code >= 500:
                if attempt < retries:
                    wait = backoff ** attempt
                    log.warning(
                        f"Server error {resp.status_code} (attempt {attempt + 1}/{retries + 1}) "
                        f"— retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                log.error(f"Server error {resp.status_code} — giving up: {resp.url}")
                resp.raise_for_status()

            if not resp.ok:
                log.error(
                    f"HTTP {resp.status_code} for {resp.url} "
                    f"— body: {resp.text[:300]}"
                )
                resp.raise_for_status()

            return resp.json()

        raise RuntimeError(f"All {retries + 1} attempts failed for {url}")

    # ── GCS backup ───────────────────────────────────────────────────────────

    def _backup_to_gcs(self, name: str, data: dict) -> None:
        """Write raw JSON to GCS for auditing and reprocessing."""
        if not (self.gcs_client and self.bucket_name):
            return
        source = self.SOURCE or "unknown"
        try:
            blob = self.gcs_client.bucket(self.bucket_name).blob(
                f"{source}/{datetime.now(timezone.utc).strftime('%Y%m%d')}/{name}.json"
            )
            blob.upload_from_string(
                json.dumps(data, default=str),
                content_type="application/json",
            )
        except Exception as exc:
            log.warning(f"GCS backup failed for {name}: {exc}")

    # ── Contract ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() is not implemented"
        )
