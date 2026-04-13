"""
scrapers/sources/spotify.py
============================
Collects playlist co-presence signal from Spotify Web API.
This is NOT a DJ set transition scraper — it builds the playlist co-occurrence
matrix used as a separate feature in the ML model.

Flow:
  1. Search for playlists matching configured queries
  2. For each playlist above min_followers threshold: fetch all tracks
  3. Write playlists + playlist_tracks to DB
  4. Backup raw JSON to GCS

Requires: spotify-client-id, spotify-client-secret in Secret Manager
API docs: https://developer.spotify.com/documentation/web-api
"""

import time
import json
import logging
import hashlib
import requests
from datetime import datetime

from utils.secrets import get_spotify_credentials
from utils.db import upsert_tracks, upsert_playlist, insert_playlist_tracks
from utils.tracklist_parser import make_track_id

log = logging.getLogger(__name__)

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE  = "https://api.spotify.com/v1"


class SpotifyScraper:
    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        self.config      = config
        self.gcs_client  = gcs_client
        self.bucket_name = bucket_name
        self._token      = None
        self._token_exp  = 0

    def _get_token(self) -> str:
        """Client credentials OAuth flow — no user login needed."""
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        client_id, client_secret = get_spotify_credentials()
        resp = requests.post(
            SPOTIFY_TOKEN_URL,
            data    = {"grant_type": "client_credentials"},
            auth    = (client_id, client_secret),
            timeout = 10,
        )
        resp.raise_for_status()
        data             = resp.json()
        self._token      = data["access_token"]
        self._token_exp  = time.time() + data["expires_in"]
        return self._token

    def _get(self, url: str, params: dict = None) -> dict:
        """Authenticated GET with rate limit handling."""
        rps   = self.config.get("requests_per_second", 3)
        delay = 1.0 / rps
        time.sleep(delay)

        headers = {"Authorization": f"Bearer {self._get_token()}"}
        resp    = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 5))
            log.warning(f"Spotify rate limit hit — waiting {retry_after}s")
            time.sleep(retry_after)
            return self._get(url, params)

        resp.raise_for_status()
        return resp.json()

    def _search_playlists(self, query: str, limit: int = 50) -> list[dict]:
        """Search for playlists matching a query."""
        results  = []
        offset   = 0
        while len(results) < limit:
            data = self._get(f"{SPOTIFY_API_BASE}/search", params={
                "q":      query,
                "type":   "playlist",
                "limit":  min(50, limit - len(results)),
                "offset": offset,
                "market": "US",
            })
            items = data.get("playlists", {}).get("items", [])
            if not items:
                break
            results.extend([i for i in items if i])  # filter None entries
            offset += len(items)
            if len(items) < 50:
                break
        return results

    def _get_playlist_tracks(self, playlist_id: str) -> list[dict]:
        """Fetch all tracks in a playlist (handles pagination)."""
        tracks = []
        url    = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks"
        params = {"limit": 100, "fields": "items(track(id,name,artists,album)),next"}

        while url:
            data  = self._get(url, params=params)
            items = data.get("items", [])
            for item in items:
                track = item.get("track")
                if not track or not track.get("id"):
                    continue
                artist = ", ".join(a["name"] for a in track.get("artists", []))
                tracks.append({
                    "spotify_id": track["id"],
                    "title":      track["name"],
                    "artist":     artist,
                })
            url    = data.get("next")
            params = {}  # next URL has params embedded

        return tracks

    def run(self) -> dict:
        """Main entry point. Returns stats dict."""
        stats = {"playlists": 0, "tracks": 0, "errors": 0}
        max_playlists = self.config.get("max_playlists_per_run", 200)
        min_followers = self.config.get("min_followers", 500)
        seen_playlist_ids = set()

        for query in self.config.get("search_queries", []):
            log.info(f"Searching Spotify playlists: '{query}'")
            try:
                playlists = self._search_playlists(query, limit=50)
            except Exception as e:
                log.error(f"Search failed for '{query}': {e}")
                stats["errors"] += 1
                continue

            for pl in playlists:
                if stats["playlists"] >= max_playlists:
                    log.info(f"Reached max_playlists_per_run={max_playlists}")
                    return stats

                pl_id = pl.get("id")
                if not pl_id or pl_id in seen_playlist_ids:
                    continue
                seen_playlist_ids.add(pl_id)

                followers = pl.get("followers", {}).get("total", 0) if pl.get("followers") else 0
                if followers < min_followers:
                    log.debug(f"Skipping low-follower playlist {pl_id} ({followers})")
                    continue

                try:
                    tracks = self._get_playlist_tracks(pl_id)
                    if len(tracks) < 5:
                        continue

                    # Backup raw to GCS
                    if self.gcs_client and self.bucket_name:
                        self._backup_to_gcs(pl_id, {"playlist": pl, "tracks": tracks})

                    # Normalise tracks → DB format
                    db_tracks = []
                    for i, t in enumerate(tracks):
                        track_id = f"sp_{t['spotify_id']}" if t.get("spotify_id") else \
                                   make_track_id(t["artist"], t["title"])
                        db_tracks.append({
                            "track_id":    track_id,
                            "title":       t["title"],
                            "artist":      t["artist"],
                            "spotify_id":  t.get("spotify_id"),
                            "source":      "spotify",
                        })

                    upsert_tracks(db_tracks)
                    upsert_playlist({
                        "playlist_id": pl_id,
                        "title":       pl.get("name", ""),
                        "source":      "spotify",
                        "source_url":  pl.get("external_urls", {}).get("spotify"),
                        "followers":   followers,
                    })
                    insert_playlist_tracks(pl_id, [
                        {"track_id": db_tracks[i]["track_id"], "position": i}
                        for i in range(len(db_tracks))
                    ])

                    stats["playlists"] += 1
                    stats["tracks"]    += len(db_tracks)
                    log.info(f"  ✓ {pl.get('name',pl_id)[:50]} — {len(tracks)} tracks")

                except Exception as e:
                    log.error(f"Error processing playlist {pl_id}: {e}")
                    stats["errors"] += 1

        return stats

    def _backup_to_gcs(self, name: str, data: dict):
        try:
            blob = self.gcs_client.bucket(self.bucket_name).blob(
                f"spotify/{datetime.utcnow().strftime('%Y%m%d')}/{name}.json"
            )
            blob.upload_from_string(json.dumps(data), content_type="application/json")
        except Exception as e:
            log.warning(f"GCS backup failed for {name}: {e}")
