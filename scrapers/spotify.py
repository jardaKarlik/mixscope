"""
scrapers/spotify.py
====================
Collects playlist co-presence signal from the Spotify Web API.

This is NOT a DJ set transition scraper — it builds the playlist
co-occurrence matrix used as a separate feature in the ML model.

Status: PARKED — tracks blocked by Spotify API policy.
  - Search works (limit ≤ 10 in dev mode)
  - GET /playlists/{id}/tracks returns 403 without Extended Quota Mode
  - GET /playlists/{id} returns metadata but strips the tracks field
  → Apply for Extended Quota Mode at developer.spotify.com/dashboard
    then revert _get_playlist_tracks() to use /tracks endpoint.

Requires: spotify-client-id, spotify-client-secret in Secret Manager
API docs: https://developer.spotify.com/documentation/web-api
"""

import time
import json
import logging
import requests
from datetime import datetime

from base import BaseScraper
from utils.secrets import get_spotify_credentials
from utils.db import upsert_tracks, upsert_playlist, insert_playlist_tracks
from utils.tracklist_parser import make_track_id

log = logging.getLogger(__name__)

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE  = "https://api.spotify.com/v1"


class SpotifyScraper(BaseScraper):
    """
    Builds playlist co-occurrence signal from Spotify.

    Config keys (scraper_config.yaml → spotify section):
        search_queries          list[str]   playlist search terms
        min_followers           int         skip playlists below this (default 500)
        max_playlists_per_run   int         hard cap (default 200)
        requests_per_second     float       rate limit (default 5)
    """

    SOURCE = "spotify"

    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        super().__init__(config, gcs_client, bucket_name)
        self._token     = None
        self._token_exp = 0

    # ── Auth ─────────────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Client Credentials OAuth — no user login needed."""
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
        data            = resp.json()
        self._token     = data["access_token"]
        self._token_exp = time.time() + data["expires_in"]
        return self._token

    # ── Spotify-specific _get (needs auth header) ────────────────────────────

    def _get(self, url: str, params: dict = None) -> dict:
        """Authenticated GET with rate-limit handling."""
        time.sleep(1.0 / self._rps)
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        resp    = requests.get(url, headers=headers, params=params or {}, timeout=15)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            log.warning(f"Spotify rate limit — waiting {wait}s")
            time.sleep(wait)
            return self._get(url, params)

        if not resp.ok:
            log.error(f"Spotify API {resp.status_code} for {resp.url} — body: {resp.text[:500]}")
            resp.raise_for_status()
        return resp.json()

    # ── Search / fetch ────────────────────────────────────────────────────────

    def _search_playlists(self, query: str, limit: int = 50) -> list[dict]:
        results = []
        offset  = 0
        while len(results) < limit:
            data  = self._get(f"{SPOTIFY_API_BASE}/search", params={
                "q":      query,
                "type":   "playlist",
                "limit":  min(10, limit - len(results)),  # dev-mode cap ~10
                "offset": offset,
                "market": "US",
            })
            items = data.get("playlists", {}).get("items", [])
            if not items:
                break
            valid = [i for i in items if i]
            results.extend(valid)
            offset += len(items)
            if len(items) < 10:
                break
        return results

    def _get_playlist_tracks(self, playlist_id: str) -> list[dict]:
        """
        Attempt to fetch tracks via GET /playlists/{id}.
        NOTE: Spotify strips the tracks field for unapproved apps.
              This will return [] until Extended Quota Mode is granted.
        """
        tracks = []
        data   = self._get(f"{SPOTIFY_API_BASE}/playlists/{playlist_id}")
        page   = data.get("tracks", {})

        for item in page.get("items", []):
            track = item.get("track")
            if not track or not track.get("id"):
                continue
            artist = ", ".join(a["name"] for a in track.get("artists", []))
            tracks.append({
                "spotify_id": track["id"],
                "title":      track["name"],
                "artist":     artist,
            })

        url = page.get("next")
        while url:
            page = self._get(url)
            for item in page.get("items", []):
                track = item.get("track")
                if not track or not track.get("id"):
                    continue
                artist = ", ".join(a["name"] for a in track.get("artists", []))
                tracks.append({
                    "spotify_id": track["id"],
                    "title":      track["name"],
                    "artist":     artist,
                })
            url = page.get("next")

        return tracks

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats             = {"playlists": 0, "tracks": 0, "errors": 0}
        max_playlists     = self.config.get("max_playlists_per_run", 200)
        min_followers     = self.config.get("min_followers", 500)
        seen_playlist_ids = set()

        for query in self.config.get("search_queries", []):
            log.info(f"Searching Spotify playlists: '{query}'")
            try:
                playlists = self._search_playlists(query, limit=50)
            except Exception as exc:
                log.error(f"Search failed for '{query}': {exc}")
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

                followers_data = pl.get("followers")
                if followers_data is not None:
                    followers = followers_data.get("total", 0)
                    if followers < min_followers:
                        continue
                else:
                    followers = 0

                try:
                    tracks = self._get_playlist_tracks(pl_id)
                    if len(tracks) < 5:
                        continue

                    self._backup_to_gcs(pl_id, {"playlist": pl, "tracks": tracks})

                    db_tracks = []
                    for i, t in enumerate(tracks):
                        track_id = f"sp_{t['spotify_id']}" if t.get("spotify_id") \
                                   else make_track_id(t["artist"], t["title"])
                        db_tracks.append({
                            "track_id":   track_id,
                            "title":      t["title"],
                            "artist":     t["artist"],
                            "spotify_id": t.get("spotify_id"),
                            "source":     "spotify",
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
                    log.info(f"  ✓ {pl.get('name', pl_id)[:50]} — {len(tracks)} tracks")

                except Exception as exc:
                    log.error(f"Error processing playlist {pl_id}: {exc}")
                    stats["errors"] += 1

        return stats
