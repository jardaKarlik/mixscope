"""
scrapers/sources/mixcloud.py
=============================
Collects DJ set tracklists from Mixcloud API.
Mixcloud is the cleanest source — tracklists are structured, not free-text.

Flow:
  1. Search for mixes by genre tags
  2. For each mix above min_play_count and min_duration: fetch tracklist
  3. Write sets + transitions to DB

API docs: https://www.mixcloud.com/developers/
No auth needed for public data (read-only). Rate limit: 1000 req/hour.
"""

import time
import json
import logging
from datetime import datetime
import requests

from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import make_track_id, tracks_to_transitions

log = logging.getLogger(__name__)

MIXCLOUD_API = "https://api.mixcloud.com"


class MixcloudScraper:
    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        self.config      = config
        self.gcs_client  = gcs_client
        self.bucket_name = bucket_name

    def _get(self, path: str, params: dict = None) -> dict:
        time.sleep(1.0 / self.config.get("requests_per_second", 2))
        resp = requests.get(f"{MIXCLOUD_API}{path}", params=params or {}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _search_mixes(self, tag: str, limit: int = 100) -> list[dict]:
        """Search mixes by genre tag."""
        mixes  = []
        offset = 0
        while len(mixes) < limit:
            data  = self._get(f"/tag/{tag}/cloudcasts/", params={
                "limit":  min(50, limit - len(mixes)),
                "offset": offset,
                "order":  "latest",
            })
            items = data.get("data", [])
            if not items:
                break
            mixes.extend(items)
            offset += len(items)
            if len(items) < 50 or not data.get("paging", {}).get("next"):
                break
        return mixes

    def _get_tracklist(self, cloudcast_key: str) -> list[dict]:
        """
        Fetch structured tracklist for a mix.
        Returns list of {artist, title, start_time}.
        Mixcloud provides this as proper structured data — no parsing needed.
        """
        try:
            data = self._get(f"{cloudcast_key}sections/", params={"limit": 100})
        except Exception:
            return []

        tracks = []
        for i, section in enumerate(data.get("data", [])):
            track = section.get("track", {})
            if not track:
                continue
            artist = track.get("artist", {}).get("name", "") or ""
            title  = track.get("name", "") or ""
            if artist and title:
                tracks.append({
                    "artist":     artist,
                    "title":      title,
                    "position":   i,
                    "start_time": section.get("start_time", 0),
                })
        return tracks

    def run(self) -> dict:
        stats    = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        max_runs = self.config.get("max_per_run", 500)
        min_plays = self.config.get("min_play_count", 2000)
        min_dur   = self.config.get("min_duration_minutes", 45) * 60

        for tag in self.config.get("genres", ["techno"]):
            if stats["sets"] >= max_runs:
                break
            log.info(f"Mixcloud searching tag: {tag}")

            try:
                mixes = self._search_mixes(tag, limit=100)
            except Exception as e:
                log.error(f"Search failed for tag {tag}: {e}")
                stats["errors"] += 1
                continue

            for mix in mixes:
                if stats["sets"] >= max_runs:
                    break

                play_count = mix.get("play_count", 0) or 0
                duration   = mix.get("audio_length", 0) or 0
                key        = mix.get("key", "")

                if play_count < min_plays:
                    continue
                if duration < min_dur:
                    continue
                if not key:
                    continue

                try:
                    raw_tracks = self._get_tracklist(key)
                    if len(raw_tracks) < 5:
                        continue

                    set_id    = f"mc_{key.strip('/').replace('/', '_')}"
                    set_date  = None
                    created   = mix.get("created_time", "")
                    if created:
                        try:
                            set_date = datetime.strptime(created[:10], "%Y-%m-%d").date()
                        except Exception:
                            pass

                    db_tracks = []
                    for t in raw_tracks:
                        track_id = make_track_id(t["artist"], t["title"])
                        db_tracks.append({**t, "track_id": track_id, "source": "mixcloud"})

                    transitions = tracks_to_transitions(db_tracks, set_id, set_date, "mixcloud")

                    if self.gcs_client and self.bucket_name:
                        self._backup_to_gcs(set_id, {"mix": mix, "tracks": raw_tracks})

                    upsert_tracks(db_tracks)
                    is_new = upsert_set({
                        "set_id":     set_id,
                        "title":      mix.get("name", "")[:255],
                        "dj":         mix.get("user", {}).get("name", ""),
                        "source":     "mixcloud",
                        "source_url": f"https://www.mixcloud.com{key}",
                        "set_date":   set_date,
                        "duration_s": duration,
                    })
                    if is_new:
                        insert_transitions(transitions)
                        stats["sets"]        += 1
                        stats["tracks"]      += len(db_tracks)
                        stats["transitions"] += len(transitions)
                        log.info(f"  ✓ {mix.get('name','')[:50]} — {len(raw_tracks)} tracks")

                except Exception as e:
                    log.error(f"Error processing mix {key}: {e}")
                    stats["errors"] += 1

        return stats

    def _backup_to_gcs(self, name: str, data: dict):
        try:
            blob = self.gcs_client.bucket(self.bucket_name).blob(
                f"mixcloud/{datetime.utcnow().strftime('%Y%m%d')}/{name}.json"
            )
            blob.upload_from_string(json.dumps(data), content_type="application/json")
        except Exception as e:
            log.warning(f"GCS backup failed {name}: {e}")
