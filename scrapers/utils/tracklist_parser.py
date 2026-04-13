"""
scrapers/utils/tracklist_parser.py
===================================
Parse raw tracklist text (from YouTube descriptions, SoundCloud descriptions,
1001Tracklists HTML, etc.) into structured (artist, title) pairs.

This is the hardest part of the pipeline — tracklist formatting is wildly
inconsistent. We try multiple patterns and pick the best parse.
"""

import re
import hashlib
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Common separators between artist and title
SEPARATORS = [" - ", " – ", " — ", " :: ", " | ", " / "]

# Patterns that indicate a line is NOT a track (timestamps, headers, etc.)
NOISE_PATTERNS = [
    r"^\d{1,2}:\d{2}(:\d{2})?$",          # bare timestamp
    r"^(tracklist|track list|playlist)[:.]?$",
    r"^(follow|subscribe|download|buy|stream|support)",
    r"^https?://",
    r"^@\w+",
    r"^\s*$",
    r"^[-—=_]{3,}",                         # divider lines
]

# Timestamp prefix pattern: "01:23:45 Artist - Title" or "01:23 Artist - Title"
TIMESTAMP_PREFIX = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s+(.+)$")

# Track number prefix: "01. Artist - Title" or "1) Artist - Title"
TRACK_NUM_PREFIX = re.compile(r"^\d{1,3}[.)]\s+(.+)$")


def _is_noise(line: str) -> bool:
    line_l = line.strip().lower()
    return any(re.match(p, line_l) for p in NOISE_PATTERNS)


def _strip_prefixes(line: str) -> str:
    """Remove timestamp and track number prefixes."""
    line = line.strip()
    m = TIMESTAMP_PREFIX.match(line)
    if m:
        line = m.group(1).strip()
    m = TRACK_NUM_PREFIX.match(line)
    if m:
        line = m.group(1).strip()
    return line


def _split_artist_title(line: str) -> Optional[tuple[str, str]]:
    """
    Split 'Artist - Title' into (artist, title).
    Returns None if no separator found.
    """
    for sep in SEPARATORS:
        if sep in line:
            parts = line.split(sep, 1)
            artist = parts[0].strip()
            title  = parts[1].strip()
            if artist and title and len(artist) > 1 and len(title) > 1:
                return artist, title
    return None


def parse_tracklist(text: str) -> list[dict]:
    """
    Parse raw tracklist text into list of dicts:
    [{ "artist": str, "title": str, "position": int, "raw_line": str }, ...]

    Returns empty list if confidence is too low.
    """
    if not text:
        return []

    lines    = text.split("\n")
    tracks   = []
    position = 0
    attempts = 0

    for line in lines:
        line = line.strip()
        if not line or _is_noise(line):
            continue

        clean  = _strip_prefixes(line)
        result = _split_artist_title(clean)
        attempts += 1

        if result:
            artist, title = result
            tracks.append({
                "artist":   _clean_name(artist),
                "title":    _clean_name(title),
                "position": position,
                "raw_line": line,
            })
            position += 1

    # Confidence: fraction of non-noise lines that parsed successfully
    confidence = len(tracks) / max(attempts, 1)
    if confidence < 0.3 and len(tracks) < 3:
        log.debug(f"Low parse confidence ({confidence:.2f}), {len(tracks)} tracks — discarding")
        return []

    log.debug(f"Parsed {len(tracks)} tracks (confidence={confidence:.2f})")
    return tracks


def _clean_name(s: str) -> str:
    """Remove common junk from artist/title names."""
    # Remove [FREE DOWNLOAD], (Original Mix), etc.
    s = re.sub(r"\[free\s+download\]", "", s, flags=re.I)
    s = re.sub(r"\(premiere\)", "", s, flags=re.I)
    # Normalise whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_track_id(artist: str, title: str) -> str:
    """
    Generate a stable track_id from artist + title.
    Used when no MusicBrainz ID is available.
    Lowercased and stripped for consistency.
    """
    key = f"{artist.lower().strip()}|{title.lower().strip()}"
    return "h_" + hashlib.md5(key.encode()).hexdigest()[:16]


def tracks_to_transitions(
    tracks: list[dict],
    set_id: str,
    set_date,
    source: str,
) -> list[dict]:
    """
    Convert ordered track list into A→B transition pairs.
    Each consecutive pair becomes one transition.
    """
    transitions = []
    for i in range(len(tracks) - 1):
        a = tracks[i]
        b = tracks[i + 1]
        transitions.append({
            "track_a_id": a["track_id"],
            "track_b_id": b["track_id"],
            "set_id":     set_id,
            "position":   a["position"],
            "set_date":   set_date,
            "source":     source,
        })
    return transitions
