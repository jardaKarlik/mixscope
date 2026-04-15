"""
scrapers/youtube.py
====================
Extracts tracklists from YouTube video descriptions.

Flow:
  1. Search channel whitelist for DJ set videos (within quota)
  2. For each video: fetch description via videos.list (1 unit each)
  3. Parse description for tracklist using tracklist_parser
  4. Write sets + transitions to DB

Quota budget (YouTube Data API v3 free tier = 10 000 units/day):
  search.list  = 100 units per call, returns up to 50 results
  videos.list  =   1 unit per call, returns up to 50 videos
  At 5 000 units/run: ~4 searches (400u) + 4 600 video lookups

Requires: youtube-api-key in GCP Secret Manager
API docs: https://developers.google.com/youtube/v3/docs
"""

import time
import logging
from datetime import datetime

import requests

from base import BaseScraper
from utils.secrets import get_youtube_api_key
from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import parse_tracklist, make_track_id, tracks_to_transitions

log = logging.getLogger(__name__)

YT_API = "https://www.googleapis.com/youtube/v3"


class YouTubeScraper(BaseScraper):
    """
    Scrapes DJ set tracklists from YouTube video descriptions.

    Config keys (scraper_config.yaml → youtube section):
        channel_whitelist           list[str]   YouTube channel IDs to search
        quota_per_run               int         API unit budget per run (default 5000)
        min_video_duration_minutes  int         skip short videos (default 45)
        min_views                   int         skip low-view videos (default 5000)
        date_from                   str         ISO date — only videos after this
        requests_per_second         float       rate limit (default 1)
    """

    SOURCE = "youtube"

    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        super().__init__(config, gcs_client, bucket_name)
        self.quota_used  = 0
        self.quota_limit = config.get("quota_per_run", 5000)

    # ── YouTube-specific _get with quota tracking ────────────────────────────

    def _get(self, endpoint: str, params: dict, cost: int = 1) -> dict:
        """
        YouTube API GET with quota tracking and rate limiting.
        Raises RuntimeError when the quota budget is exhausted.
        """
        if self.quota_used + cost > self.quota_limit:
            raise RuntimeError(
                f"YouTube quota limit reached ({self.quota_limit} units)"
            )

        time.sleep(1.0 / self._rps)
        params["key"] = get_youtube_api_key()

        resp = requests.get(f"{YT_API}/{endpoint}", params=params, timeout=15)

        if resp.status_code == 403:
            body = resp.json().get("error", {}).get("message", resp.text[:200])
            log.error(f"YouTube API 403: {body}")
            raise RuntimeError("YouTube API 403 — quota exceeded or key invalid")

        if not resp.ok:
            log.error(f"YouTube API {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()

        self.quota_used += cost
        return resp.json()

    # ── Search / detail fetchers ─────────────────────────────────────────────

    def _search_channel(self, channel_id: str, query: str = "DJ set") -> list[str]:
        """Search a channel for DJ set videos. Returns list of video IDs."""
        video_ids  = []
        page_token = None

        while True:
            params = {
                "part":           "id",
                "channelId":      channel_id,
                "q":              query,
                "type":           "video",
                "videoDuration":  "long",   # > 20 min — pre-filters shorts
                "maxResults":     50,
                "order":          "date",
                "publishedAfter": self.config.get("date_from", "2019-01-01") + "T00:00:00Z",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._get("search", params, cost=100)
            except RuntimeError as exc:
                log.warning(f"Quota/error during channel search {channel_id}: {exc}")
                break

            for item in data.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    video_ids.append(vid)

            page_token = data.get("nextPageToken")
            if not page_token or len(video_ids) >= 200:
                break

        return video_ids

    def _get_video_details(self, video_ids: list[str]) -> list[dict]:
        """
        Fetch video details in batches of 50 (1 quota unit per batch).
        Returns list of raw video item dicts.
        """
        results = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            try:
                data = self._get("videos", {
                    "part": "snippet,contentDetails,statistics",
                    "id":   ",".join(batch),
                }, cost=1)
            except RuntimeError:
                break
            results.extend(data.get("items", []))
        return results

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_duration_seconds(iso_duration: str) -> int:
        try:
            from isodate import parse_duration
            return int(parse_duration(iso_duration).total_seconds())
        except Exception:
            return 0

    @staticmethod
    def _extract_set_date(published_at: str):
        try:
            return datetime.strptime(published_at[:10], "%Y-%m-%d").date()
        except Exception:
            return None

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats     = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        min_dur_s = self.config.get("min_video_duration_minutes", 45) * 60
        min_views = self.config.get("min_views", 5000)
        channels  = self.config.get("channel_whitelist", [])

        log.info(
            f"YouTube scraper: {len(channels)} channel(s), "
            f"quota={self.quota_limit}"
        )

        for channel_id in channels:
            if self.quota_used >= self.quota_limit:
                log.info("Quota exhausted — stopping.")
                break

            log.info(f"Searching channel {channel_id} …")
            try:
                video_ids = self._search_channel(channel_id)
            except Exception as exc:
                log.error(f"Channel search failed {channel_id}: {exc}")
                stats["errors"] += 1
                continue

            log.info(f"  Found {len(video_ids)} candidate videos")
            if not video_ids:
                continue

            try:
                videos = self._get_video_details(video_ids)
            except Exception as exc:
                log.error(f"Video details fetch failed: {exc}")
                stats["errors"] += 1
                continue

            for video in videos:
                snippet       = video.get("snippet", {})
                details       = video.get("contentDetails", {})
                statistics    = video.get("statistics", {})
                video_id      = video["id"]
                duration_s    = self._parse_duration_seconds(details.get("duration", "PT0S"))
                view_count    = int(statistics.get("viewCount", 0))
                description   = snippet.get("description", "")
                title         = snippet.get("title", "")
                published_at  = snippet.get("publishedAt", "")
                set_date      = self._extract_set_date(published_at)
                channel_title = snippet.get("channelTitle", "")

                if duration_s < min_dur_s:
                    log.debug(f"  Skip short video {video_id} ({duration_s}s)")
                    continue
                if view_count < min_views:
                    log.debug(f"  Skip low-view video {video_id} ({view_count} views)")
                    continue

                raw_tracks = parse_tracklist(description)
                if len(raw_tracks) < 5:
                    log.debug(f"  No tracklist in {video_id} ({len(raw_tracks)} parsed)")
                    continue

                self._backup_to_gcs(video_id, {
                    "video_id":    video_id,
                    "title":       title,
                    "description": description,
                    "tracks":      raw_tracks,
                })

                set_id    = f"yt_{video_id}"
                db_tracks = []
                for t in raw_tracks:
                    track_id = make_track_id(t["artist"], t["title"])
                    db_tracks.append({**t, "track_id": track_id, "source": "youtube"})

                transitions = tracks_to_transitions(db_tracks, set_id, set_date, "youtube")

                try:
                    upsert_tracks(db_tracks)
                    is_new = upsert_set({
                        "set_id":     set_id,
                        "title":      title[:255],
                        "dj":         channel_title,
                        "source":     "youtube",
                        "source_url": f"https://youtube.com/watch?v={video_id}",
                        "set_date":   set_date,
                        "duration_s": duration_s,
                    })
                    if is_new:
                        insert_transitions(transitions)
                        stats["sets"]        += 1
                        stats["tracks"]      += len(db_tracks)
                        stats["transitions"] += len(transitions)
                        log.info(
                            f"  ✓ {title[:60]} — "
                            f"{len(raw_tracks)} tracks, "
                            f"{len(transitions)} transitions"
                        )
                except Exception as exc:
                    log.error(f"DB write failed for {video_id}: {exc}")
                    stats["errors"] += 1

        log.info(
            f"YouTube done. Quota used: {self.quota_used}/{self.quota_limit}. "
            f"Stats: {stats}"
        )
        return stats
