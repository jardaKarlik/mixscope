"""
scrapers/resident_advisor.py
=============================
Collects DJ mix tracklists from Resident Advisor (ra.co/podcasts).

RA's GraphQL API (ra.co/graphql) is open — no authentication required.
The `podcast` type has a `tracklist` field with plain-text "Artist - Title"
lines, available for ~83% of episodes. Average 20-40 tracks per episode.

Data model:
  podcasts(limit: 10)          → discover the latest DB ID
  podcast(id: X) { ... }       → fetch individual episodes by ID

Pagination strategy:
  IDs are sequential integers (RA.1053 = DB id 1053 as of 2026-04).
  `podcasts(limit: N)` returns 0 for N > 10 (server-side cap).
  So we: 1) fetch 10 most recent to find max_id, then 2) walk backward
  via individual podcast(id: X) calls until date_from or max_per_run.

Subsequent runs are efficient: stop as soon as we hit an already-known
episode (upsert_set returns False → already in DB).

Rate: no published limit; 1 req/s is empirically safe.
No API key or Secret Manager required.
"""

import time
import logging
import re
from datetime import datetime, date

import requests

from base import BaseScraper
from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import make_track_id, tracks_to_transitions, parse_tracklist

log = logging.getLogger(__name__)

RA_GQL = "https://ra.co/graphql"

# ── GraphQL fragments used by both queries ───────────────────────────────────

_PODCAST_FIELDS = """
    id
    date
    title
    duration
    tracklist
    contentUrl
    artist {
      id
      name
      urlSafeName
    }
"""

_QUERY_LATEST = f"""
{{
  podcasts(limit: 10) {{
    {_PODCAST_FIELDS}
  }}
}}
"""

_QUERY_BY_ID = """
query PodcastById($id: ID!) {{
  podcast(id: $id) {{
    {fields}
  }}
}}
"""


def _query_by_id(podcast_id: int) -> str:
    return f"""
    {{
      podcast(id: {podcast_id}) {{
        {_PODCAST_FIELDS}
      }}
    }}
    """


# ── Duration parsing ─────────────────────────────────────────────────────────

def _parse_duration_s(duration_str: str | None) -> int | None:
    """
    Parse RA duration string 'HH:MM:SS' or 'MM:SS' to seconds.
    Returns None if unparseable.
    """
    if not duration_str:
        return None
    parts = duration_str.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


# ── HTML entity cleanup ──────────────────────────────────────────────────────

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
}


def _clean_tracklist_text(text: str) -> str:
    """Strip any stray HTML and decode entities from tracklist text."""
    text = re.sub(r"<[^>]+>", " ", text)  # remove tags
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return text


class ResidentAdvisorScraper(BaseScraper):
    """
    Scrapes DJ mix tracklists from Resident Advisor podcasts via GraphQL.

    Config keys (scraper_config.yaml → residentadvisor section):
        date_from            str    ISO date — only episodes on/after this (default "2018-01-01")
        max_per_run          int    hard cap on new sets stored per run (default 200)
        requests_per_second  float  rate limit (default 1)
        min_tracklist_length int    skip episodes with fewer parsed tracks (default 5)
        stop_on_known        bool   stop walking backward once a known episode is seen (default True)
                                    Set False on first run if you want to backfill fully.
    """

    SOURCE = "residentadvisor"

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Content-Type": "application/json",
        "Accept":       "application/json",
        "Referer":      "https://ra.co/podcasts",
        "Origin":       "https://ra.co",
    }

    # ── GraphQL helper ────────────────────────────────────────────────────────

    def _gql(self, query: str, variables: dict = None) -> dict:
        """
        POST a GraphQL query to RA. Rate-limited; raises on HTTP error.
        Returns the parsed JSON dict (may contain 'errors' key).
        """
        time.sleep(1.0 / max(self._rps, 0.1))
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        resp = requests.post(RA_GQL, json=payload, headers=self._HEADERS, timeout=20)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 30))
            log.warning(f"RA rate limit — waiting {wait}s")
            time.sleep(wait)
            return self._gql(query, variables)

        if not resp.ok:
            log.error(f"RA GraphQL {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()

        return resp.json()

    # ── Fetchers ─────────────────────────────────────────────────────────────

    def _get_latest_episodes(self) -> list[dict]:
        """Return the 10 most recent podcast episodes (RA's server-side cap)."""
        data = self._gql(_QUERY_LATEST)
        return (data.get("data") or {}).get("podcasts") or []

    def _fetch_episode(self, podcast_id: int) -> dict | None:
        """
        Fetch a single podcast episode by its DB ID.
        Returns the episode dict or None if not found.
        """
        data = self._gql(_query_by_id(podcast_id))
        return (data.get("data") or {}).get("podcast")

    # ── Tracklist extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_tracklist(raw_tracklist: str) -> list[dict]:
        """
        Parse the RA tracklist text into structured track list.
        RA format is plain-text "Artist - Title" lines (no HTML, no timestamps).

        Returns [{artist, title, position, start_time}, ...]
        """
        if not raw_tracklist:
            return []
        cleaned = _clean_tracklist_text(raw_tracklist)
        parsed  = parse_tracklist(cleaned)
        return [
            {
                "artist":     t["artist"],
                "title":      t["title"],
                "position":   t["position"],
                "start_time": 0,
            }
            for t in parsed
        ]

    # ── Episode → DB ─────────────────────────────────────────────────────────

    def _process_episode(self, ep: dict, min_tracks: int) -> str | None:
        """
        Validate, parse, and write a single episode to the DB.

        Returns:
          "new"    — written successfully (new row)
          "known"  — already in DB (skip signal for pagination)
          "skip"   — filtered out (short tracklist, missing data, etc.)
          "error"  — exception during DB write
        """
        ep_id   = ep.get("id")
        title   = (ep.get("title") or "").strip()[:255]
        raw_tl  = ep.get("tracklist") or ""
        artist  = ep.get("artist") or {}
        dj      = (artist.get("name") or "").strip()
        url     = ep.get("contentUrl") or f"https://ra.co/podcasts/{ep_id}"
        dur_str = ep.get("duration")
        created = ep.get("date") or ""

        if not ep_id or not title:
            log.debug(f"  Skip episode {ep_id} — missing id/title")
            return "skip"

        # Parse tracklist
        raw_tracks = self._extract_tracklist(raw_tl)
        if len(raw_tracks) < min_tracks:
            log.debug(
                f"  Skip RA.{ep_id} '{title[:40]}' — "
                f"only {len(raw_tracks)} tracks parsed (min {min_tracks})"
            )
            return "skip"

        # Parse date
        set_date = None
        if created:
            try:
                set_date = datetime.strptime(created[:10], "%Y-%m-%d").date()
            except ValueError:
                pass

        duration_s = _parse_duration_s(dur_str)
        set_id     = f"ra_{ep_id}"

        # Build DB objects — assign track_ids first (needed for transitions)
        db_tracks = []
        for t in raw_tracks:
            track_id = make_track_id(t["artist"], t["title"])
            db_tracks.append({
                **t,
                "track_id": track_id,
                "source":   "residentadvisor",
            })

        transitions = tracks_to_transitions(db_tracks, set_id, set_date, "residentadvisor")

        # Deduplicate by track_id before upsert — the same track can appear
        # multiple times in a set (two plays of the same record). PostgreSQL
        # raises "cannot affect row a second time" if duplicates are in one
        # execute_values batch with ON CONFLICT DO UPDATE.
        seen: set[str] = set()
        unique_tracks  = []
        for t in db_tracks:
            if t["track_id"] not in seen:
                seen.add(t["track_id"])
                unique_tracks.append(t)

        # GCS backup
        self._backup_to_gcs(set_id, {
            "episode":   ep,
            "raw_tracks": raw_tracks,
        })

        try:
            upsert_tracks(unique_tracks)
            is_new = upsert_set({
                "set_id":     set_id,
                "title":      title,
                "dj":         dj,
                "source":     "residentadvisor",
                "source_url": url,
                "set_date":   set_date,
                "duration_s": duration_s,
            })
            if is_new:
                insert_transitions(transitions)
                log.info(
                    f"  + RA.{ep_id}  {title[:55]}  [{dj}] — "
                    f"{len(raw_tracks)} tracks, "
                    f"{len(transitions)} transitions"
                )
                return "new"
            else:
                log.debug(f"  ~ RA.{ep_id} already in DB: '{title}'")
                return "known"
        except Exception as exc:
            log.error(f"  ! DB write failed for RA.{ep_id} '{title}': {exc}")
            return "error"

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats         = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        max_sets      = self.config.get("max_per_run", 200)
        date_from_str = self.config.get("date_from", "2018-01-01")
        min_tracks    = self.config.get("min_tracklist_length", 5)
        stop_on_known = self.config.get("stop_on_known", True)

        try:
            date_from = datetime.strptime(date_from_str, "%Y-%m-%d").date()
        except ValueError:
            log.warning(f"Invalid date_from '{date_from_str}' — defaulting to 2018-01-01")
            date_from = date(2018, 1, 1)

        log.info(
            f"RA scraper starting — "
            f"date_from={date_from}, max_per_run={max_sets}, "
            f"min_tracks={min_tracks}, stop_on_known={stop_on_known}"
        )

        # ── Phase 1: Discover the latest episode ID ───────────────────────────
        log.info("Fetching latest episodes to find current max ID…")
        try:
            latest = self._get_latest_episodes()
        except Exception as exc:
            log.error(f"Failed to fetch latest episodes: {exc}")
            stats["errors"] += 1
            return stats

        if not latest:
            log.error("No episodes returned from RA — API may have changed.")
            stats["errors"] += 1
            return stats

        max_id  = max(int(ep["id"]) for ep in latest)
        min_id  = min(int(ep["id"]) for ep in latest)
        log.info(f"  Latest episodes: IDs {min_id}–{max_id} ({len(latest)} total)")

        # ── Phase 2: Process the latest batch first ───────────────────────────
        # Sort newest-first so we process recent episodes before walking backward
        latest_sorted = sorted(latest, key=lambda e: int(e["id"]), reverse=True)
        consecutive_known = 0

        for ep in latest_sorted:
            if stats["sets"] >= max_sets:
                break
            ep_date_str = (ep.get("date") or "")[:10]
            try:
                ep_date = datetime.strptime(ep_date_str, "%Y-%m-%d").date() if ep_date_str else None
            except ValueError:
                ep_date = None

            if ep_date and ep_date < date_from:
                log.info(f"  Reached date_from={date_from} — stopping.")
                return stats

            result = self._process_episode(ep, min_tracks)
            if result == "new":
                stats["sets"]        += 1
                # Approximate track/transition counts from what was written
                tl = self._extract_tracklist(ep.get("tracklist") or "")
                stats["tracks"]      += len(tl)
                stats["transitions"] += max(0, len(tl) - 1)
                consecutive_known = 0
            elif result == "known":
                consecutive_known += 1
                if stop_on_known and consecutive_known >= 3:
                    log.info("  3 consecutive known episodes — likely up to date, stopping.")
                    return stats
            elif result == "error":
                stats["errors"] += 1

        # ── Phase 3: Walk backward by ID ────────────────────────────────────
        current_id        = min_id - 1
        consecutive_known = 0
        consecutive_miss  = 0  # IDs that return null (gaps in the series)

        log.info(f"Walking backward from ID {current_id}…")

        while stats["sets"] < max_sets:
            if current_id < 1:
                log.info("Reached ID=1 — all historical episodes processed.")
                break

            try:
                ep = self._fetch_episode(current_id)
            except Exception as exc:
                log.warning(f"  Fetch failed for ID {current_id}: {exc}")
                stats["errors"] += 1
                current_id -= 1
                continue

            current_id -= 1

            if ep is None:
                consecutive_miss += 1
                if consecutive_miss >= 10:
                    log.info(f"  10 consecutive null IDs — series may have ended. Stopping.")
                    break
                continue
            consecutive_miss = 0

            # Date check
            ep_date_str = (ep.get("date") or "")[:10]
            try:
                ep_date = datetime.strptime(ep_date_str, "%Y-%m-%d").date() if ep_date_str else None
            except ValueError:
                ep_date = None

            if ep_date and ep_date < date_from:
                log.info(f"  Reached date_from={date_from} at RA.{ep.get('id')} — stopping.")
                break

            result = self._process_episode(ep, min_tracks)
            if result == "new":
                stats["sets"]        += 1
                tl = self._extract_tracklist(ep.get("tracklist") or "")
                stats["tracks"]      += len(tl)
                stats["transitions"] += max(0, len(tl) - 1)
                consecutive_known = 0
            elif result == "known":
                consecutive_known += 1
                if stop_on_known and consecutive_known >= 5:
                    log.info(
                        f"  5 consecutive known episodes — "
                        f"already scraped this range. Stopping."
                    )
                    break
            elif result == "error":
                stats["errors"] += 1

        log.info(
            f"RA done — "
            f"sets={stats['sets']}, tracks={stats['tracks']}, "
            f"transitions={stats['transitions']}, errors={stats['errors']}"
        )
        return stats
