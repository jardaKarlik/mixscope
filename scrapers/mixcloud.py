"""
scrapers/mixcloud.py
====================
Collects DJ set tracklists from the Mixcloud API.

Mixcloud is the cleanest source — tracklists are structured data,
not free-text descriptions, so no parsing is needed.

API:  https://www.mixcloud.com/developers/
Auth: none required for public read-only data
Rate: 1 000 req/hour  →  keep requests_per_second ≤ 2 in config
"""

import logging
from datetime import datetime

from base import BaseScraper
from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import make_track_id, tracks_to_transitions

log = logging.getLogger(__name__)

MIXCLOUD_API = "https://api.mixcloud.com"


class MixcloudScraper(BaseScraper):
    """
    Scrapes DJ mix tracklists from Mixcloud by genre tag.

    Config keys (from scraper_config.yaml → mixcloud section):
        genres              list[str]   genre tags to search
        min_play_count      int         skip mixes below this threshold
        min_duration_minutes int        skip mixes shorter than this (default 45)
        max_per_run         int         hard cap on sets stored per run
        requests_per_second float       rate limit (default 2)
    """

    SOURCE = "mixcloud"

    # ── API helpers ──────────────────────────────────────────────────────────

    def _api_get(self, path: str, params: dict = None) -> dict:
        """GET against the Mixcloud API base URL."""
        return self._get(f"{MIXCLOUD_API}{path}", params=params)

    def _search_mixes(self, tag: str, limit: int = 100) -> list[dict]:
        """
        Return up to `limit` cloudcast objects for a genre tag,
        ordered by most recent first.
        """
        mixes  = []
        offset = 0

        while len(mixes) < limit:
            batch_size = min(50, limit - len(mixes))
            data = self._api_get(f"/tag/{tag}/cloudcasts/", params={
                "limit":  batch_size,
                "offset": offset,
                "order":  "latest",
            })
            items = data.get("data", [])
            if not items:
                break
            mixes.extend(items)
            offset += len(items)
            # Stop if Mixcloud signals no more pages
            if len(items) < batch_size or not data.get("paging", {}).get("next"):
                break

        return mixes

    def _get_tracklist(self, cloudcast_key: str) -> list[dict]:
        """
        Fetch structured tracklist for a mix via the /sections/ endpoint.

        Returns list of dicts:
            artist     str
            title      str
            position   int   (0-based index in set)
            start_time int   (seconds from mix start)

        Returns [] on any error — caller skips mixes with < 5 tracks.
        """
        try:
            data = self._api_get(f"{cloudcast_key}sections/", params={"limit": 100})
        except Exception as exc:
            log.debug(f"Tracklist fetch failed for {cloudcast_key}: {exc}")
            return []

        tracks = []
        for i, section in enumerate(data.get("data", [])):
            track = section.get("track") or {}
            artist = (track.get("artist") or {}).get("name", "").strip()
            title  = (track.get("name") or "").strip()
            if artist and title:
                tracks.append({
                    "artist":     artist,
                    "title":      title,
                    "position":   i,
                    "start_time": section.get("start_time", 0),
                })

        return tracks

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats     = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        max_sets  = self.config.get("max_per_run", 500)
        min_plays = self.config.get("min_play_count", 2000)
        min_dur_s = self.config.get("min_duration_minutes", 45) * 60
        genres    = self.config.get("genres", ["techno"])

        log.info(
            f"Mixcloud scraper starting — "
            f"{len(genres)} genre(s), max_sets={max_sets}, "
            f"min_plays={min_plays}, min_dur={min_dur_s // 60}min"
        )

        for tag in genres:
            if stats["sets"] >= max_sets:
                log.info(f"Reached max_per_run={max_sets} — stopping.")
                break

            log.info(f"Searching tag: '{tag}' …")
            try:
                mixes = self._search_mixes(tag, limit=100)
            except Exception as exc:
                log.error(f"Search failed for tag '{tag}': {exc}")
                stats["errors"] += 1
                continue

            log.info(f"  Found {len(mixes)} candidate mixes for '{tag}'")

            for mix in mixes:
                if stats["sets"] >= max_sets:
                    break

                # ── Filter ──────────────────────────────────────────────────
                play_count = mix.get("play_count") or 0
                duration_s = mix.get("audio_length") or 0
                key        = mix.get("key", "")
                name       = mix.get("name", "")[:255]
                dj         = (mix.get("user") or {}).get("name", "")

                if not key:
                    continue
                if play_count < min_plays:
                    log.debug(f"  Skip low-play mix '{name}' ({play_count} plays)")
                    continue
                if duration_s < min_dur_s:
                    log.debug(f"  Skip short mix '{name}' ({duration_s // 60}min)")
                    continue

                # ── Fetch tracklist ──────────────────────────────────────────
                try:
                    raw_tracks = self._get_tracklist(key)
                    if len(raw_tracks) < 5:
                        log.debug(f"  Skip mix with too few tracks '{name}' ({len(raw_tracks)})")
                        continue

                    # Parse set date
                    set_date = None
                    created  = mix.get("created_time", "")
                    if created:
                        try:
                            set_date = datetime.strptime(created[:10], "%Y-%m-%d").date()
                        except ValueError:
                            pass

                    # Build DB objects
                    set_id    = f"mc_{key.strip('/').replace('/', '_')}"
                    db_tracks = []
                    for t in raw_tracks:
                        track_id = make_track_id(t["artist"], t["title"])
                        db_tracks.append({
                            **t,
                            "track_id": track_id,
                            "source":   "mixcloud",
                        })

                    transitions = tracks_to_transitions(db_tracks, set_id, set_date, "mixcloud")

                    # ── GCS backup ───────────────────────────────────────────
                    self._backup_to_gcs(set_id, {"mix": mix, "tracks": raw_tracks})

                    # ── Write to DB ──────────────────────────────────────────
                    upsert_tracks(db_tracks)
                    is_new = upsert_set({
                        "set_id":     set_id,
                        "title":      name,
                        "dj":         dj,
                        "source":     "mixcloud",
                        "source_url": f"https://www.mixcloud.com{key}",
                        "set_date":   set_date,
                        "duration_s": duration_s,
                    })
                    if is_new:
                        insert_transitions(transitions)
                        stats["sets"]        += 1
                        stats["tracks"]      += len(db_tracks)
                        stats["transitions"] += len(transitions)
                        log.info(
                            f"  ✓ {name[:60]} "
                            f"[{dj}] — "
                            f"{len(raw_tracks)} tracks, "
                            f"{len(transitions)} transitions, "
                            f"{play_count:,} plays"
                        )
                    else:
                        log.debug(f"  Already in DB: '{name}'")

                except Exception as exc:
                    log.error(f"Error processing mix '{name}' ({key}): {exc}")
                    stats["errors"] += 1

        log.info(
            f"Mixcloud done — "
            f"sets={stats['sets']}, tracks={stats['tracks']}, "
            f"transitions={stats['transitions']}, errors={stats['errors']}"
        )
        return stats
