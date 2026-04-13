"""
scrapers/sources/youtube.py
============================
Extracts tracklists from YouTube video descriptions.

Flow:
  1. Search channel whitelist for DJ set videos (within quota)
  2. For each video: fetch description via videos.list (1 unit each)
  3. Parse description for tracklist using tracklist_parser
  4. Write sets + transitions to DB

Quota budget:
  search.list   = 100 units per call, returns up to 50 results
  videos.list   = 1 unit per call, returns up to 50 videos
  At 5000 units/run: ~4 searches (400u) + 4600 video lookups

Requires: youtube-api-key in Secret Manager
API docs: https://developers.google.com/youtube/v3/docs
"""

import time
import json
import logging
import hashlib
import re
from datetime import datetime
from isodate import parse_duration
import requests

from utils.secrets import get_youtube_api_key
from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import parse_tracklist, make_track_id, tracks_to_transitions

log = logging.getLogger(__name__)

YT_API = "https://www.googleapis.com/youtube/v3"


class YouTubeScraper:
    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        self.config       = config
        self.gcs_client   = gcs_client
        self.bucket_name  = bucket_name
        self.quota_used   = 0
        self.quota_limit  = config.get("quota_per_run", 5000)

    def _get(self, endpoint: str, params: dict, cost: int = 1) -> dict:
        """API call with quota tracking and rate limiting."""
        if self.quota_used + cost > self.quota_limit:
            raise RuntimeError(f"YouTube quota limit reached ({self.quota_limit} units)")

        time.sleep(1.0 / self.config.get("requests_per_second", 1))
        api_key = get_youtube_api_key()
        params["key"] = api_key

        resp = requests.get(f"{YT_API}/{endpoint}", params=params, timeout=15)
        if resp.status_code == 403:
            log.error("YouTube API quota exceeded or key invalid")
            raise RuntimeError("YouTube API 403")
        resp.raise_for_status()
        self.quota_used += cost
        return resp.json()

    def _search_channel(self, channel_id: str, query: str = "DJ set") -> list[str]:
        """Search a channel for DJ set videos. Returns list of video IDs."""
        video_ids = []
        page_token = None

        while True:
            params = {
                "part":       "id",
                "channelId":  channel_id,
                "q":          query,
                "type":       "video",
                "videoDuration": "long",  # >20 min — pre-filters shorts
                "maxResults": 50,
                "order":      "date",
                "publishedAfter": self.config.get("date_from", "2019-01-01") + "T00:00:00Z",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._get("search", params, cost=100)
            except RuntimeError as e:
                log.warning(f"Quota/error during search: {e}")
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
        Fetch video details in batches of 50 (1 unit per batch).
        Returns list of video dicts with description, duration, title, etc.
        """
        results = []
        for i in range(0, len(video_ids), 50):
            batch  = video_ids[i:i+50]
            try:
                data = self._get("videos", {
                    "part": "snippet,contentDetails,statistics",
                    "id":   ",".join(batch),
                }, cost=1)
            except RuntimeError:
                break
            results.extend(data.get("items", []))
        return results

    def _parse_duration_seconds(self, iso_duration: str) -> int:
        try:
            return int(parse_duration(iso_duration).total_seconds())
        except Exception:
            return 0

    def _extract_set_date(self, published_at: str):
        try:
            return datetime.strptime(published_at[:10], "%Y-%m-%d").date()
        except Exception:
            return None

    def run(self) -> dict:
        """Main entry point."""
        stats    = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        min_dur  = self.config.get("min_video_duration_minutes", 45) * 60
        min_views = self.config.get("min_views", 5000)

        channel_whitelist = self.config.get("channel_whitelist", [])
        log.info(f"YouTube scraper: {len(channel_whitelist)} channels, quota={self.quota_limit}")

        for channel_id in channel_whitelist:
            if self.quota_used >= self.quota_limit:
                break

            log.info(f"Searching channel {channel_id}...")
            try:
                video_ids = self._search_channel(channel_id)
            except Exception as e:
                log.error(f"Channel search failed {channel_id}: {e}")
                stats["errors"] += 1
                continue

            log.info(f"  Found {len(video_ids)} candidate videos")
            if not video_ids:
                continue

            try:
                videos = self._get_video_details(video_ids)
            except Exception as e:
                log.error(f"Video details failed: {e}")
                stats["errors"] += 1
                continue

            for video in videos:
                snippet        = video.get("snippet", {})
                details        = video.get("contentDetails", {})
                statistics     = video.get("statistics", {})
                video_id       = video["id"]
                duration_s     = self._parse_duration_seconds(details.get("duration","PT0S"))
                view_count     = int(statistics.get("viewCount", 0))
                description    = snippet.get("description", "")
                title          = snippet.get("title", "")
                published_at   = snippet.get("publishedAt", "")
                set_date       = self._extract_set_date(published_at)
                channel_title  = snippet.get("channelTitle", "")

                # Apply filters
                if duration_s < min_dur:
                    log.debug(f"Skip short video {video_id} ({duration_s}s)")
                    continue
                if view_count < min_views:
                    log.debug(f"Skip low-view video {video_id} ({view_count} views)")
                    continue

                # Parse tracklist from description
                raw_tracks = parse_tracklist(description)
                if len(raw_tracks) < 5:
                    log.debug(f"No tracklist in {video_id} ({len(raw_tracks)} parsed)")
                    continue

                # GCS backup
                if self.gcs_client and self.bucket_name:
                    self._backup_to_gcs(video_id, {
                        "video_id": video_id, "title": title,
                        "description": description, "tracks": raw_tracks
                    })

                # Build track objects
                set_id    = f"yt_{video_id}"
                db_tracks = []
                for t in raw_tracks:
                    track_id = make_track_id(t["artist"], t["title"])
                    db_tracks.append({
                        **t,
                        "track_id": track_id,
                        "source":   "youtube",
                    })

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
                        log.info(f"  ✓ {title[:60]} — {len(raw_tracks)} tracks, {len(transitions)} transitions")
                except Exception as e:
                    log.error(f"DB write failed for {video_id}: {e}")
                    stats["errors"] += 1

        log.info(f"YouTube done. Quota used: {self.quota_used}/{self.quota_limit}. Stats: {stats}")
        return stats

    def _backup_to_gcs(self, name: str, data: dict):
        try:
            blob = self.gcs_client.bucket(self.bucket_name).blob(
                f"youtube/{datetime.utcnow().strftime('%Y%m%d')}/{name}.json"
            )
            blob.upload_from_string(json.dumps(data), content_type="application/json")
        except Exception as e:
            log.warning(f"GCS backup failed {name}: {e}")
