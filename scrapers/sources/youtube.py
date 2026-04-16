"""
scrapers/sources/youtube.py
============================
Extracts tracklists from YouTube video descriptions.

Sources:
  1. Whitelist channels — scraped exhaustively, source_quality=1
  2. Search queries   — broader discovery, source_quality=2 (tracklist found)
                        or source_quality=3 (no tracklist / Option-D fallback)

Quota budget (YouTube Data API v3, 10k units/day free):
  search.list  = 100 units per call, up to 50 results
  videos.list  =   1 unit per batch of 50

@-handle channel IDs are resolved to UC*** IDs via channels.list (1 unit each)
and cached for the lifetime of the scraper run.

Requires: youtube-api-key in Secret Manager
API docs: https://developers.google.com/youtube/v3/docs
"""

import re
import time
import json
import logging
from datetime import datetime
from isodate import parse_duration
import requests

from utils.secrets import get_youtube_api_key
from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import parse_tracklist, make_track_id, tracks_to_transitions

log = logging.getLogger(__name__)

YT_API = "https://www.googleapis.com/youtube/v3"


def has_tracklist(description: str) -> bool:
    """Return True if description contains tracklist markers (timestamps or numbered lines)."""
    if not description:
        return False
    if re.search(r"^\s*\d{1,2}:\d{2}(:\d{2})?", description, re.MULTILINE):
        return True
    if re.search(r"^\s*\d{1,2}\.\s+\S", description, re.MULTILINE):
        return True
    if re.search(r"\btracklist\b", description, re.IGNORECASE):
        return True
    return False


class YouTubeScraper:
    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        self.config       = config
        self.gcs_client   = gcs_client
        self.bucket_name  = bucket_name
        self.quota_used   = 0
        self.quota_limit  = config.get("quota_per_run", 5000)
        self._handle_cache: dict[str, str] = {}  # @handle → UC*** id

    # ─── API helpers ──────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict, cost: int = 1) -> dict:
        if self.quota_used + cost > self.quota_limit:
            raise RuntimeError(f"YouTube quota limit reached ({self.quota_limit} units)")
        time.sleep(1.0 / self.config.get("requests_per_second", 1))
        params["key"] = get_youtube_api_key()
        resp = requests.get(f"{YT_API}/{endpoint}", params=params, timeout=15)
        if resp.status_code == 403:
            log.error("YouTube API quota exceeded or key invalid")
            raise RuntimeError("YouTube API 403")
        resp.raise_for_status()
        self.quota_used += cost
        return resp.json()

    def _resolve_channel_id(self, handle_or_id: str) -> str:
        """Resolve a @handle to a UC*** channel ID. Returns input unchanged if already UC***."""
        if not handle_or_id.startswith("@"):
            return handle_or_id
        if handle_or_id in self._handle_cache:
            return self._handle_cache[handle_or_id]
        try:
            data = self._get("channels", {"part": "id", "forHandle": handle_or_id.lstrip("@")}, cost=1)
            items = data.get("items", [])
            if items:
                resolved = items[0]["id"]
                self._handle_cache[handle_or_id] = resolved
                log.info(f"Resolved {handle_or_id} → {resolved}")
                return resolved
        except Exception as e:
            log.warning(f"Could not resolve channel handle {handle_or_id}: {e}")
        self._handle_cache[handle_or_id] = handle_or_id  # cache failure to avoid retries
        return handle_or_id

    def _title_is_excluded(self, title: str) -> bool:
        """Return True if title matches any exclusion keyword."""
        lower = title.lower()
        for kw in self.config.get("title_exclude_keywords", []):
            if kw.lower() in lower:
                return True
        return False

    def _search_channel(self, channel_id: str) -> list[str]:
        """Search a channel for DJ set videos. Returns list of video IDs."""
        video_ids  = []
        page_token = None
        date_from  = self.config.get("date_from")

        while True:
            params = {
                "part":          "id",
                "channelId":     channel_id,
                "q":             "DJ set",
                "type":          "video",
                "videoDuration": "long",
                "maxResults":    50,
                "order":         "date",
            }
            if date_from:
                params["publishedAfter"] = f"{date_from}T00:00:00Z"
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._get("search", params, cost=100)
            except RuntimeError as e:
                log.warning(f"Quota/error during channel search: {e}")
                break

            for item in data.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    video_ids.append(vid)

            page_token = data.get("nextPageToken")
            if not page_token or len(video_ids) >= 200:
                break

        return video_ids

    def _search_query(self, query: str) -> list[str]:
        """Search globally for a query. Returns up to 50 video IDs."""
        params = {
            "part":          "id",
            "q":             query,
            "type":          "video",
            "videoDuration": "long",
            "maxResults":    50,
            "order":         "relevance",
        }
        date_from = self.config.get("date_from")
        if date_from:
            params["publishedAfter"] = f"{date_from}T00:00:00Z"

        try:
            data = self._get("search", params, cost=100)
        except RuntimeError as e:
            log.warning(f"Quota/error during query search '{query}': {e}")
            return []

        return [
            item.get("id", {}).get("videoId")
            for item in data.get("items", [])
            if item.get("id", {}).get("videoId")
        ]

    def _get_video_details(self, video_ids: list[str]) -> list[dict]:
        """Fetch video details in batches of 50 (1 unit per batch)."""
        results = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
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

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def _ingest_videos(
        self,
        video_ids: list[str],
        whitelist_channel_ids: set[str],
        stats: dict,
    ) -> None:
        """Fetch details for video_ids and ingest qualifying ones."""
        if not video_ids:
            return

        min_dur   = self.config.get("min_video_duration_minutes", 45) * 60
        min_views = self.config.get("min_views", 100)
        videos    = self._get_video_details(video_ids)

        for video in videos:
            snippet    = video.get("snippet", {})
            details    = video.get("contentDetails", {})
            statistics = video.get("statistics", {})

            video_id      = video["id"]
            title         = snippet.get("title", "")
            channel_id    = snippet.get("channelId", "")
            channel_title = snippet.get("channelTitle", "")
            published_at  = snippet.get("publishedAt", "")
            description   = snippet.get("description", "")
            duration_s    = self._parse_duration_seconds(details.get("duration", "PT0S"))
            view_count    = int(statistics.get("viewCount", 0))
            set_date      = self._extract_set_date(published_at)

            if duration_s < min_dur:
                log.debug(f"Skip short video {video_id} ({duration_s}s)")
                continue
            if view_count < min_views:
                log.debug(f"Skip low-view video {video_id} ({view_count} views)")
                continue
            if self._title_is_excluded(title):
                log.debug(f"Skip excluded title: {title!r}")
                continue

            # ── Determine source_quality ───────────────────────────────────
            if channel_id in whitelist_channel_ids:
                source_quality = 1
            elif has_tracklist(description):
                source_quality = 2
            else:
                source_quality = 3

            # Parse tracklist from description
            raw_tracks = parse_tracklist(description)
            if len(raw_tracks) < 5:
                log.debug(f"No tracklist in {video_id} ({len(raw_tracks)} parsed)")
                continue

            if self.gcs_client and self.bucket_name:
                self._backup_to_gcs(video_id, {
                    "video_id": video_id, "title": title,
                    "description": description, "tracks": raw_tracks,
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
                    "set_id":         set_id,
                    "title":          title[:255],
                    "dj":             channel_title,
                    "source":         "youtube",
                    "source_url":     f"https://youtube.com/watch?v={video_id}",
                    "set_date":       set_date,
                    "duration_s":     duration_s,
                    "source_quality": source_quality,
                })
                if is_new:
                    insert_transitions(transitions)
                    stats["sets"]        += 1
                    stats["tracks"]      += len(db_tracks)
                    stats["transitions"] += len(transitions)
                    q_label = {1: "whitelist", 2: "search+tl", 3: "search+notl"}[source_quality]
                    log.info(f"  ✓ [{q_label}] {title[:60]} — {len(raw_tracks)} tracks")
            except Exception as e:
                log.error(f"DB write failed for {video_id}: {e}")
                stats["errors"] += 1

    # ─── Main entry point ─────────────────────────────────────────────────────

    def run(self) -> dict:
        stats = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}

        # Resolve all channel handles → UC*** IDs up front
        raw_whitelist = self.config.get("channel_whitelist", [])
        whitelist_ids = set()
        for entry in raw_whitelist:
            resolved = self._resolve_channel_id(entry)
            if resolved:
                whitelist_ids.add(resolved)

        log.info(f"YouTube scraper: {len(whitelist_ids)} channels, "
                 f"{len(self.config.get('search_queries', []))} queries, "
                 f"quota={self.quota_limit}")

        # ── Phase 1: Whitelist channels (source_quality=1) ─────────────────
        for channel_id in whitelist_ids:
            if self.quota_used >= self.quota_limit:
                log.info("Quota limit reached, stopping whitelist phase.")
                break
            log.info(f"Searching whitelist channel {channel_id}...")
            try:
                video_ids = self._search_channel(channel_id)
            except Exception as e:
                log.error(f"Channel search failed {channel_id}: {e}")
                stats["errors"] += 1
                continue
            log.info(f"  Found {len(video_ids)} candidate videos")
            self._ingest_videos(video_ids, whitelist_ids, stats)

        # ── Phase 2: Search queries (source_quality=2 or 3) ───────────────
        seen_video_ids: set[str] = set()
        for query in self.config.get("search_queries", []):
            if self.quota_used >= self.quota_limit:
                log.info("Quota limit reached, stopping search phase.")
                break
            log.info(f"Searching query: '{query}'")
            try:
                video_ids = self._search_query(query)
            except Exception as e:
                log.error(f"Search failed for '{query}': {e}")
                stats["errors"] += 1
                continue
            # Deduplicate across queries and skip already-processed whitelist videos
            new_ids = [vid for vid in video_ids if vid not in seen_video_ids]
            seen_video_ids.update(new_ids)
            log.info(f"  {len(new_ids)} new candidate videos")
            self._ingest_videos(new_ids, whitelist_ids, stats)

        log.info(
            f"YouTube done. Quota used: {self.quota_used}/{self.quota_limit}. "
            f"Sets: {stats['sets']}, Tracks: {stats['tracks']}, "
            f"Transitions: {stats['transitions']}, Errors: {stats['errors']}"
        )
        return stats

    def _backup_to_gcs(self, name: str, data: dict):
        try:
            blob = self.gcs_client.bucket(self.bucket_name).blob(
                f"youtube/{datetime.utcnow().strftime('%Y%m%d')}/{name}.json"
            )
            blob.upload_from_string(json.dumps(data), content_type="application/json")
        except Exception as e:
            log.warning(f"GCS backup failed {name}: {e}")
