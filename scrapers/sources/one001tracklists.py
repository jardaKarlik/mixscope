"""
scrapers/sources/one001tracklists.py
=====================================
HTML scraper for 1001Tracklists.com — the highest quality transition source.
Structured tracklist data, professionally maintained.

Note: No official API. Uses respectful rate limiting (0.5 req/s).
      Be a good citizen — don't hammer their servers.

Flow:
  1. Browse genre pages for recent tracklists
  2. For each tracklist page: parse structured track listings
  3. Write sets + transitions to DB
"""

import time
import json
import logging
import re
import hashlib
from datetime import datetime
from typing import Optional
import requests
from bs4 import BeautifulSoup

from utils.db import upsert_tracks, upsert_set, insert_transitions
from utils.tracklist_parser import make_track_id, tracks_to_transitions

log = logging.getLogger(__name__)

BASE_URL = "https://www.1001tracklists.com"


class OneZeroZeroOneTracklists:
    def __init__(self, config: dict, gcs_client=None, bucket_name: str = None):
        self.config      = config
        self.gcs_client  = gcs_client
        self.bucket_name = bucket_name
        self.session     = requests.Session()
        self.session.headers.update({
            "User-Agent": "MixscopeBot/1.0 (DJ mix research; github.com/JardaKarlik/mixscope)"
        })

    def _get(self, url: str) -> Optional[BeautifulSoup]:
        delay = 1.0 / self.config.get("requests_per_second", 0.5)
        time.sleep(delay)
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            log.error(f"GET failed {url}: {e}")
            return None

    def _get_tracklist_urls(self, genre: str, page: int = 1) -> list[str]:
        """Get tracklist URLs from a genre listing page."""
        url  = f"{BASE_URL}/genre/{genre}/page/{page}/"
        soup = self._get(url)
        if not soup:
            return []
        links = soup.select("a[href*='/tracklist/']")
        return list(set(BASE_URL + a["href"] for a in links if a.get("href")))

    def _parse_tracklist_page(self, url: str) -> Optional[dict]:
        """
        Parse a single tracklist page.
        Returns dict with set metadata and ordered tracks.
        """
        soup = self._get(url)
        if not soup:
            return None

        # Extract set title and DJ
        title_el = soup.select_one("h1.tlTitle, h1.title")
        title    = title_el.get_text(strip=True) if title_el else ""

        # Extract date
        date_el = soup.select_one("[class*='date'], time[datetime]")
        set_date = None
        if date_el:
            dt_str = date_el.get("datetime") or date_el.get_text(strip=True)
            for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d.%m.%Y"):
                try:
                    set_date = datetime.strptime(dt_str[:10], fmt).date()
                    break
                except Exception:
                    pass

        # Extract tracks — 1001TL uses structured markup
        tracks  = []
        track_items = soup.select(".tlpItemContainer, .tlpItem, [class*='trackItem']")
        for i, item in enumerate(track_items):
            artist_el = item.select_one(".tlpArtist, .artistName, [class*='artist']")
            title_el  = item.select_one(".tlpTitle, .trackTitle, [class*='title']")
            if not artist_el or not title_el:
                continue
            artist = artist_el.get_text(strip=True)
            title  = title_el.get_text(strip=True)
            if artist and title:
                tracks.append({"artist": artist, "title": title, "position": i})

        if len(tracks) < self.config.get("min_tracklist_length", 10):
            return None

        return {"title": title, "set_date": set_date, "url": url, "tracks": tracks}

    def run(self) -> dict:
        stats   = {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
        max_run = self.config.get("max_per_run", 500)

        for genre in self.config.get("genres", ["techno"]):
            if stats["sets"] >= max_run:
                break
            log.info(f"1001Tracklists scanning genre: {genre}")

            for page in range(1, 10):
                if stats["sets"] >= max_run:
                    break
                urls = self._get_tracklist_urls(genre, page)
                if not urls:
                    break

                for url in urls:
                    if stats["sets"] >= max_run:
                        break
                    try:
                        result = self._parse_tracklist_page(url)
                        if not result:
                            continue

                        set_id   = "ot_" + hashlib.md5(url.encode()).hexdigest()[:12]
                        db_tracks = []
                        for t in result["tracks"]:
                            track_id = make_track_id(t["artist"], t["title"])
                            db_tracks.append({**t, "track_id": track_id, "source": "1001tracklists"})

                        transitions = tracks_to_transitions(
                            db_tracks, set_id, result["set_date"], "1001tracklists"
                        )
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
                            log.info(f"  ✓ {result['title'][:50]} — {len(db_tracks)} tracks")

                    except Exception as e:
                        log.error(f"Error processing {url}: {e}")
                        stats["errors"] += 1

        return stats
