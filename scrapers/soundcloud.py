"""
scrapers/soundcloud.py
=======================
Collects DJ mix tracklists from SoundCloud.

SoundCloud API situation:
  - Public API v1 (api.soundcloud.com) closed to new apps since 2019.
  - Internal API v2 (api-v2.soundcloud.com) is what the SC web app uses.
    It requires a `client_id` that is embedded in SoundCloud's JS bundle.
  - client_id is NOT a secret (visible to any browser) but does rotate
    when SoundCloud deploys new JS. We handle this by:
      1. Checking Secret Manager for "soundcloud-client-id" (optional cache).
      2. Falling back to extracting it live from SoundCloud's homepage JS.
      3. Re-extracting automatically on any 401 (rotation detected).

Flow:
  1. Obtain client_id (Secret Manager → JS extraction)
  2. For each search query: paginate /search/tracks results
  3. Filter by duration, play count, public status
  4. Parse tracklist from description field via tracklist_parser
  5. Write sets + transitions to DB

To cache the extracted client_id in Secret Manager (saves a JS fetch per run):
  gcloud secrets create mixscope-soundcloud-client-id \\
      --replication-policy=automatic --project=mixsource
  echo -n "<extracted_id>" | \\
      gcloud secrets versions add mixscope-soundcloud-client-id --data-file=-

Rate limits: SoundCloud doesn't publish limits; ~2 req/s is empirically safe.
API base:    https://api-v2.soundcloud.com
"""

import re
import time
import logging
from datetime import datetime

import requests

from base import BaseScraper
from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import make_track_id, tracks_to_transitions, parse_tracklist

log = logging.getLogger(__name__)

SC_API        = "https://api-v2.soundcloud.com"
SC_WEB        = "https://soundcloud.com"
SC_CDN        = "https://a-v2.sndcdn.com"

# Regex to extract client_id from a minified JS bundle
_CLIENT_ID_RE = re.compile(
    r'client_id\s*[=:]\s*["\']([A-Za-z0-9]{20,})["\']'
)

# SoundCloud tag_list uses space-separated tokens; multi-word tags are quoted:
# '"hard techno" techno rave "dj set"'
_TAG_RE = re.compile(r'"([^"]+)"|(\S+)')


def _parse_tag_list(tag_list: str) -> list[str]:
    """Parse SoundCloud tag_list into a clean list of strings."""
    if not tag_list:
        return []
    return [m.group(1) or m.group(2) for m in _TAG_RE.finditer(tag_list)]


class SoundCloudScraper(BaseScraper):
    """
    Scrapes DJ mix tracklists from SoundCloud via the internal API v2.

    Config keys (scraper_config.yaml → soundcloud section):
        search_queries       list[str]   free-text search queries to run
        min_play_count       int         skip tracks below this (default 5000)
        min_duration_minutes int         skip tracks shorter than this (default 45)
        max_per_run          int         hard cap on sets stored per run (default 300)
        requests_per_second  float       rate limit — keep ≤ 2 (default 2)
    """

    SOURCE = "soundcloud"

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        super().__init__(config, gcs_client, bucket_name)
        self._client_id: str | None = None

    # ── client_id management ─────────────────────────────────────────────────

    def _load_client_id(self) -> str:
        """
        Obtain a valid SoundCloud client_id.

        Priority:
          1. Already cached in self._client_id.
          2. Secret Manager (mixscope-soundcloud-client-id) — optional cache.
          3. Live extraction from SoundCloud homepage JS bundle.

        Raises RuntimeError if extraction fails.
        """
        if self._client_id:
            return self._client_id

        # ── Try Secret Manager ───────────────────────────────────────────────
        try:
            from utils.secrets import get_secret
            self._client_id = get_secret("soundcloud-client-id")
            log.info("SoundCloud client_id loaded from Secret Manager.")
            return self._client_id
        except Exception as exc:
            log.info(f"Secret Manager miss for soundcloud-client-id ({exc}) — extracting from JS.")

        # ── Extract from JS bundle ───────────────────────────────────────────
        self._client_id = self._extract_client_id_from_js()
        log.info(f"SoundCloud client_id extracted from JS: {self._client_id[:8]}…")
        return self._client_id

    def _extract_client_id_from_js(self) -> str:
        """
        Fetch SoundCloud homepage, find JS bundle URLs, scan for client_id.
        Returns the client_id string or raises RuntimeError.
        """
        time.sleep(1.0 / max(self._rps, 0.01))
        resp = requests.get(SC_WEB, headers=self._HEADERS, timeout=15)
        resp.raise_for_status()

        # Find all JS bundle <script src="..."> references
        js_urls = re.findall(
            r'src="(https://a-v2\.sndcdn\.com/assets/[^"]+\.js)"',
            resp.text,
        )
        if not js_urls:
            raise RuntimeError("No SC JS bundles found on homepage — site structure changed?")

        log.debug(f"Found {len(js_urls)} JS bundles to scan for client_id")

        for url in js_urls:
            try:
                time.sleep(0.2)
                js_resp = requests.get(url, headers=self._HEADERS, timeout=10)
                if not js_resp.ok:
                    continue
                m = _CLIENT_ID_RE.search(js_resp.text)
                if m:
                    return m.group(1)
            except Exception as exc:
                log.debug(f"JS bundle fetch failed {url}: {exc}")

        raise RuntimeError(
            "Could not extract SoundCloud client_id from any JS bundle. "
            "SC may have changed their JS structure — try updating the regex."
        )

    # ── HTTP (overrides BaseScraper._get to inject client_id + rotate on 401) ─

    def _api_get(self, endpoint: str, params: dict = None, _retry_auth: bool = True) -> dict:
        """
        Authenticated GET against api-v2.soundcloud.com.

        Adds client_id to every request. On 401 (client_id rotated):
        forces re-extraction and retries exactly once.
        """
        time.sleep(1.0 / max(self._rps, 0.01))

        merged = dict(params or {})
        merged["client_id"] = self._load_client_id()

        url  = endpoint if endpoint.startswith("http") else f"{SC_API}{endpoint}"
        resp = requests.get(url, headers=self._HEADERS, params=merged, timeout=15)

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            log.warning(f"SoundCloud rate limit — waiting {wait}s")
            time.sleep(wait)
            return self._api_get(endpoint, params, _retry_auth)

        if resp.status_code == 401 and _retry_auth:
            log.warning("SoundCloud 401 — client_id expired, re-extracting from JS…")
            self._client_id = None  # force re-extraction
            try:
                self._client_id = self._extract_client_id_from_js()
                log.info(f"New client_id extracted: {self._client_id[:8]}…")
            except Exception as exc:
                raise RuntimeError(f"client_id re-extraction failed: {exc}") from exc
            return self._api_get(endpoint, params, _retry_auth=False)

        if not resp.ok:
            log.error(
                f"SoundCloud API {resp.status_code} for {resp.url} "
                f"— body: {resp.text[:300]}"
            )
            resp.raise_for_status()

        return resp.json()

    # ── Search ───────────────────────────────────────────────────────────────

    def _search_tracks(self, query: str, limit: int = 200) -> list[dict]:
        """
        Search for tracks matching query. Returns up to `limit` track dicts.
        Paginates via next_href.
        """
        tracks     = []
        next_href  = None
        page_limit = min(50, limit)

        while len(tracks) < limit:
            if next_href:
                data = self._api_get(next_href)
            else:
                data = self._api_get("/search/tracks", {
                    "q":      query,
                    "limit":  page_limit,
                    "linked_partitioning": 1,
                })

            batch = data.get("collection", [])
            if not batch:
                break

            tracks.extend(batch)
            next_href = data.get("next_href")
            if not next_href or len(batch) < page_limit:
                break

        return tracks[:limit]

    # ── Tracklist extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_tracklist(description: str) -> list[dict]:
        """
        Parse description text into structured track list.
        Returns list of {artist, title, position, start_time}.
        """
        parsed = parse_tracklist(description)
        return [
            {
                "artist":     t["artist"],
                "title":      t["title"],
                "position":   t["position"],
                "start_time": 0,   # SC descriptions rarely include timestamps
            }
            for t in parsed
        ]

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats        = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        max_sets     = self.config.get("max_per_run", 300)
        min_plays    = self.config.get("min_play_count", 5000)
        min_dur_ms   = self.config.get("min_duration_minutes", 45) * 60 * 1000
        queries      = self.config.get("search_queries", ["techno dj set tracklist"])
        seen_ids     = set()

        log.info(
            f"SoundCloud scraper starting — "
            f"{len(queries)} query/queries, max_sets={max_sets}, "
            f"min_plays={min_plays:,}, min_dur={min_dur_ms//60000}min"
        )

        # Warm up client_id once before the loop so failures are visible early
        try:
            self._load_client_id()
        except RuntimeError as exc:
            log.error(f"Cannot obtain SoundCloud client_id: {exc}")
            stats["errors"] += 1
            return stats

        for query in queries:
            if stats["sets"] >= max_sets:
                log.info(f"Reached max_per_run={max_sets} — stopping.")
                break

            log.info(f"Searching: '{query}' …")
            try:
                candidates = self._search_tracks(query, limit=200)
            except Exception as exc:
                log.error(f"Search failed for '{query}': {exc}")
                stats["errors"] += 1
                continue

            log.info(f"  {len(candidates)} candidates for '{query}'")

            for track in candidates:
                if stats["sets"] >= max_sets:
                    break

                track_id   = track.get("id")
                title      = (track.get("title") or "").strip()[:255]
                duration   = track.get("full_duration") or track.get("duration") or 0
                plays      = track.get("playback_count") or 0
                genre      = (track.get("genre") or "").strip()
                tag_list   = track.get("tag_list") or ""
                description = track.get("description") or ""
                permalink  = track.get("permalink_url") or ""
                created_at = track.get("created_at") or ""
                user       = track.get("user") or {}
                dj         = (user.get("username") or "").strip()

                if not track_id or not permalink:
                    continue
                if track_id in seen_ids:
                    continue
                seen_ids.add(track_id)

                # ── Filters ──────────────────────────────────────────────────
                if track.get("sharing") != "public":
                    log.debug(f"  Skip non-public track {track_id}")
                    continue
                if duration < min_dur_ms:
                    log.debug(
                        f"  Skip short track '{title}' "
                        f"({duration//60000}min < {min_dur_ms//60000}min)"
                    )
                    continue
                if plays < min_plays:
                    log.debug(f"  Skip low-play track '{title}' ({plays:,} plays)")
                    continue

                # ── Extract tracklist from description ────────────────────────
                raw_tracks = self._extract_tracklist(description)
                if len(raw_tracks) < 5:
                    log.debug(
                        f"  Skip '{title}' — too few tracklist lines "
                        f"({len(raw_tracks)} parsed)"
                    )
                    continue

                # ── Parse metadata ────────────────────────────────────────────
                set_date = None
                if created_at:
                    try:
                        set_date = datetime.strptime(
                            created_at[:10], "%Y-%m-%d"
                        ).date()
                    except ValueError:
                        pass

                tags_clean = ", ".join(_parse_tag_list(tag_list)) if tag_list else None
                set_id     = f"sc_{track_id}"

                # ── Build DB objects ──────────────────────────────────────────
                db_tracks = []
                for t in raw_tracks:
                    tid = make_track_id(t["artist"], t["title"])
                    db_tracks.append({
                        "track_id":  tid,
                        "title":     t["title"],
                        "artist":    t["artist"],
                        "genre":     genre or None,
                        "lastfm_tags": tags_clean,
                        "source":    "soundcloud",
                    })

                transitions = tracks_to_transitions(
                    db_tracks, set_id, set_date, "soundcloud"
                )

                # ── GCS backup ────────────────────────────────────────────────
                self._backup_to_gcs(set_id, {
                    "track":       track,
                    "description": description,
                    "raw_tracks":  raw_tracks,
                })

                # ── Write to DB ───────────────────────────────────────────────
                try:
                    upsert_tracks(db_tracks)
                    is_new = upsert_set({
                        "set_id":     set_id,
                        "title":      title,
                        "dj":         dj,
                        "source":     "soundcloud",
                        "source_url": permalink,
                        "set_date":   set_date,
                        "duration_s": duration // 1000,
                    })
                    if is_new:
                        insert_transitions(transitions)
                        stats["sets"]        += 1
                        stats["tracks"]      += len(db_tracks)
                        stats["transitions"] += len(transitions)
                        log.info(
                            f"  ✓ {title[:60]} [{dj}] — "
                            f"{len(raw_tracks)} tracks, "
                            f"{len(transitions)} transitions, "
                            f"{plays:,} plays"
                        )
                    else:
                        log.debug(f"  Already in DB: '{title}'")

                except Exception as exc:
                    log.error(f"DB write failed for '{title}' ({set_id}): {exc}")
                    stats["errors"] += 1

        log.info(
            f"SoundCloud done — "
            f"sets={stats['sets']}, tracks={stats['tracks']}, "
            f"transitions={stats['transitions']}, errors={stats['errors']}"
        )
        return stats
