"""
scrapers/tracklists_1001.py
============================
HTML scraper for 1001Tracklists.com — the highest quality transition source.
Structured tracklist data, professionally maintained.

Note: No official API. Uses respectful rate limiting (0.5 req/s).
      Be a good citizen — don't hammer their servers.

Flow:
  1. Browse genre pages for recent tracklists
  2. For each tracklist page: parse structured track listings
  3. Write sets + transitions to DB

Requires: no API key — public site
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from base import BaseScraper
from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import make_track_id, tracks_to_transitions

log = logging.getLogger(__name__)

BASE_URL = "https://www.1001tracklists.com"


class Tracklists1001Scraper(BaseScraper):
    """
    Scrapes DJ set tracklists from 1001Tracklists.com by genre.

    Config keys (scraper_config.yaml → tracklists_1001 section):
        genres                  list[str]   genre slugs to browse
        min_tracklist_length    int         skip sets below this track count (default 10)
        max_per_run             int         hard cap on sets stored per run (default 500)
        requests_per_second     float       rate limit — keep ≤ 0.5 (default 0.5)
    """

    SOURCE = "1001tracklists"

    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        super().__init__(config, gcs_client, bucket_name)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "MixscopeBot/1.0 (DJ mix research; github.com/JardaKarlik/mixscope)"
        })

    # ── HTML helpers (site-specific — don't use the JSON _get from base) ────

    def _html_get(self, url: str) -> Optional[BeautifulSoup]:
        """Rate-limited HTML GET. Returns BeautifulSoup or None on error."""
        import time
        time.sleep(1.0 / max(self._rps, 0.01))
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            log.error(f"GET failed {url}: {exc}")
            return None

    # ── Scraping logic ───────────────────────────────────────────────────────

    def _get_tracklist_urls(self, genre: str, page: int = 1) -> list[str]:
        """Return tracklist URLs from a genre listing page."""
        url  = f"{BASE_URL}/genre/{genre}/page/{page}/"
        soup = self._html_get(url)
        if not soup:
            return []
        links = soup.select("a[href*='/tracklist/']")
        return list(set(BASE_URL + a["href"] for a in links if a.get("href")))

    def _parse_tracklist_page(self, url: str) -> Optional[dict]:
        """
        Parse a single tracklist page.
        Returns dict with set metadata + ordered tracks, or None if unusable.
        """
        soup = self._html_get(url)
        if not soup:
            return None

        # Set title
        title_el = soup.select_one("h1.tlTitle, h1.title")
        title    = title_el.get_text(strip=True) if title_el else ""

        # Set date
        date_el  = soup.select_one("[class*='date'], time[datetime]")
        set_date = None
        if date_el:
            dt_str = date_el.get("datetime") or date_el.get_text(strip=True)
            for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d.%m.%Y"):
                try:
                    set_date = datetime.strptime(dt_str[:10], fmt).date()
                    break
                except Exception:
                    pass

        # Tracks — 1001TL uses structured markup
        tracks      = []
        track_items = soup.select(".tlpItemContainer, .tlpItem, [class*='trackItem']")
        for i, item in enumerate(track_items):
            artist_el = item.select_one(".tlpArtist, .artistName, [class*='artist']")
            ttitle_el = item.select_one(".tlpTitle, .trackTitle, [class*='title']")
            if not artist_el or not ttitle_el:
                continue
            artist = artist_el.get_text(strip=True)
            ttitle = ttitle_el.get_text(strip=True)
            if artist and ttitle:
                tracks.append({"artist": artist, "title": ttitle, "position": i})

        min_len = self.config.get("min_tracklist_length", 10)
        if len(tracks) < min_len:
            log.debug(f"  Skip short tracklist '{title}' ({len(tracks)} tracks < {min_len})")
            return None

        return {"title": title, "set_date": set_date, "url": url, "tracks": tracks}

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        stats   = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        max_run = self.config.get("max_per_run", 500)
        genres  = self.config.get("genres", ["techno"])

        log.info(
            f"1001Tracklists scraper starting — "
            f"{len(genres)} genre(s), max_per_run={max_run}"
        )

        for genre in genres:
            if stats["sets"] >= max_run:
                log.info(f"Reached max_per_run={max_run} — stopping.")
                break

            log.info(f"Scanning genre: '{genre}' …")

            for page in range(1, 10):
                if stats["sets"] >= max_run:
                    break

                urls = self._get_tracklist_urls(genre, page)
                if not urls:
                    log.debug(f"  No URLs on page {page} for '{genre}' — done with genre.")
                    break

                log.info(f"  Page {page}: {len(urls)} tracklist URLs")

                for url in urls:
                    if stats["sets"] >= max_run:
                        break
                    try:
                        result = self._parse_tracklist_page(url)
                        if not result:
                            continue

                        set_id    = "ot_" + hashlib.md5(url.encode()).hexdigest()[:12]
                        db_tracks = []
                        for t in result["tracks"]:
                            track_id = make_track_id(t["artist"], t["title"])
                            db_tracks.append({
                                **t,
                                "track_id": track_id,
                                "source":   "1001tracklists",
                            })

                        transitions = tracks_to_transitions(
                            db_tracks, set_id, result["set_date"], "1001tracklists"
                        )

                        self._backup_to_gcs(set_id, {
                            "url":    url,
                            "title":  result["title"],
                            "tracks": result["tracks"],
                        })

                        upsert_tracks(db_tracks)
                        is_new = upsert_set({
                            "set_id":     set_id,
                            "title":      result["title"][:255],
                            "dj":         "",
                            "source":     "1001tracklists",
                            "source_url": url,
                            "set_date":   result["set_date"],
                            "duration_s": None,
                        })
                        if is_new:
                            insert_transitions(transitions)
                            stats["sets"]        += 1
                            stats["tracks"]      += len(db_tracks)
                            stats["transitions"] += len(transitions)
                            log.info(
                                f"  ✓ {result['title'][:60]} — "
                                f"{len(db_tracks)} tracks, "
                                f"{len(transitions)} transitions"
                            )
                        else:
                            log.debug(f"  Already in DB: '{result['title']}'")

                    except Exception as exc:
                        log.error(f"Error processing {url}: {exc}")
                        stats["errors"] += 1

        log.info(
            f"1001Tracklists done — "
            f"sets={stats['sets']}, tracks={stats['tracks']}, "
            f"transitions={stats['transitions']}, errors={stats['errors']}"
        )
        return stats
