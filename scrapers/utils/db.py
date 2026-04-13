"""
scrapers/utils/db.py
====================
PostgreSQL connection via Cloud SQL Auth Proxy (sidecar in Cloud Run).
The proxy runs on localhost:5432 inside the container — no direct DB exposure.

Schema is created here on first run via create_tables().
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from utils.secrets import get_db_password

log = logging.getLogger(__name__)

DB_NAME = os.environ.get("DB_NAME", "mixscope")
DB_USER = os.environ.get("DB_USER", "scraper")
DB_HOST = "127.0.0.1"   # Cloud SQL Auth Proxy always on localhost
DB_PORT = 5432


def get_connection():
    """Open a new psycopg2 connection. Caller is responsible for closing."""
    return psycopg2.connect(
        host     = DB_HOST,
        port     = DB_PORT,
        dbname   = DB_NAME,
        user     = DB_USER,
        password = get_db_password(),
        connect_timeout = 10,
    )


@contextmanager
def db_cursor(commit: bool = True):
    """Context manager: yields a cursor, auto-commits or rolls back."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema ───────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
-- All scraped tracks, deduplicated by mbid or (artist+title) hash
CREATE TABLE IF NOT EXISTS tracks (
    track_id        TEXT PRIMARY KEY,          -- MusicBrainz ID or generated hash
    title           TEXT NOT NULL,
    artist          TEXT NOT NULL,
    bpm             NUMERIC(6,2),
    camelot_key     TEXT,
    genre           TEXT,
    label           TEXT,
    lastfm_tags     TEXT,                      -- comma-separated
    spotify_id      TEXT,
    musicbrainz_id  TEXT,
    source          TEXT NOT NULL,             -- which scraper added it
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS tracks_artist_title ON tracks(artist, title);
CREATE INDEX IF NOT EXISTS tracks_bpm ON tracks(bpm);

-- DJ set metadata
CREATE TABLE IF NOT EXISTS sets (
    set_id      TEXT PRIMARY KEY,
    title       TEXT,
    dj          TEXT,
    source      TEXT NOT NULL,               -- youtube|mixcloud|soundcloud|1001tracklists|ra
    source_url  TEXT UNIQUE,
    set_date    DATE,
    duration_s  INTEGER,
    scraped_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Ordered transitions within DJ sets — the core training signal
CREATE TABLE IF NOT EXISTS transitions (
    id          BIGSERIAL PRIMARY KEY,
    track_a_id  TEXT NOT NULL REFERENCES tracks(track_id),
    track_b_id  TEXT NOT NULL REFERENCES tracks(track_id),
    set_id      TEXT NOT NULL REFERENCES sets(set_id),
    position    INTEGER,                     -- position of track_a in the set
    set_date    DATE,
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(track_a_id, track_b_id, set_id)  -- no duplicate transitions per set
);
CREATE INDEX IF NOT EXISTS transitions_a ON transitions(track_a_id);
CREATE INDEX IF NOT EXISTS transitions_b ON transitions(track_b_id);
CREATE INDEX IF NOT EXISTS transitions_date ON transitions(set_date);

-- Playlists (Spotify, YouTube playlists — different from DJ sets)
CREATE TABLE IF NOT EXISTS playlists (
    playlist_id TEXT PRIMARY KEY,
    title       TEXT,
    source      TEXT NOT NULL,
    source_url  TEXT,
    followers   INTEGER,
    scraped_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Track positions within playlists — for co-presence signal
CREATE TABLE IF NOT EXISTS playlist_tracks (
    playlist_id TEXT NOT NULL REFERENCES playlists(playlist_id),
    track_id    TEXT NOT NULL REFERENCES tracks(track_id),
    position    INTEGER NOT NULL,
    PRIMARY KEY (playlist_id, track_id)
);
CREATE INDEX IF NOT EXISTS pt_track ON playlist_tracks(track_id);

-- Scraper run log — track what was collected each run
CREATE TABLE IF NOT EXISTS scraper_runs (
    id               BIGSERIAL PRIMARY KEY,
    source           TEXT NOT NULL,
    started_at       TIMESTAMPTZ DEFAULT NOW(),
    finished_at      TIMESTAMPTZ,
    tracks_added     INTEGER DEFAULT 0,
    transitions_added INTEGER DEFAULT 0,
    playlists_added  INTEGER DEFAULT 0,
    errors           INTEGER DEFAULT 0,
    status           TEXT DEFAULT 'running',  -- running|success|failed
    notes            TEXT
);
"""


def create_tables():
    """Create all tables if they don't exist. Safe to run repeatedly."""
    with db_cursor() as cur:
        cur.execute(CREATE_TABLES_SQL)
    log.info("Database tables created/verified.")


# ─── Write helpers ────────────────────────────────────────────────────────────

def upsert_tracks(tracks: list[dict]) -> int:
    """
    Insert or update tracks. Returns count of new rows inserted.
    tracks: list of dicts with keys matching the tracks table columns.
    """
    if not tracks:
        return 0
    cols = ["track_id","title","artist","bpm","camelot_key","genre",
            "label","lastfm_tags","spotify_id","musicbrainz_id","source"]
    rows = [[t.get(c) for c in cols] for t in tracks]
    sql  = f"""
        INSERT INTO tracks ({','.join(cols)})
        VALUES %s
        ON CONFLICT (track_id) DO UPDATE SET
            bpm         = EXCLUDED.bpm,
            camelot_key = EXCLUDED.camelot_key,
            genre       = EXCLUDED.genre,
            label       = EXCLUDED.label,
            lastfm_tags = EXCLUDED.lastfm_tags,
            spotify_id  = EXCLUDED.spotify_id,
            updated_at  = NOW()
    """
    with db_cursor() as cur:
        execute_values(cur, sql, rows)
        return cur.rowcount


def upsert_set(set_data: dict) -> bool:
    """Insert a DJ set, skip if source_url already seen. Returns True if new."""
    sql = """
        INSERT INTO sets (set_id, title, dj, source, source_url, set_date, duration_s)
        VALUES (%(set_id)s, %(title)s, %(dj)s, %(source)s, %(source_url)s,
                %(set_date)s, %(duration_s)s)
        ON CONFLICT (source_url) DO NOTHING
        RETURNING set_id
    """
    with db_cursor() as cur:
        cur.execute(sql, set_data)
        return cur.fetchone() is not None


def insert_transitions(transitions: list[dict]) -> int:
    """Bulk insert transitions, skip duplicates. Returns rows inserted."""
    if not transitions:
        return 0
    cols = ["track_a_id","track_b_id","set_id","position","set_date","source"]
    rows = [[t.get(c) for c in cols] for t in transitions]
    sql  = f"""
        INSERT INTO transitions ({','.join(cols)})
        VALUES %s
        ON CONFLICT (track_a_id, track_b_id, set_id) DO NOTHING
    """
    with db_cursor() as cur:
        execute_values(cur, sql, rows)
        return cur.rowcount


def upsert_playlist(playlist: dict) -> bool:
    """Insert playlist, skip if already exists. Returns True if new."""
    sql = """
        INSERT INTO playlists (playlist_id, title, source, source_url, followers)
        VALUES (%(playlist_id)s, %(title)s, %(source)s, %(source_url)s, %(followers)s)
        ON CONFLICT (playlist_id) DO NOTHING
        RETURNING playlist_id
    """
    with db_cursor() as cur:
        cur.execute(sql, playlist)
        return cur.fetchone() is not None


def insert_playlist_tracks(playlist_id: str, tracks: list[dict]) -> int:
    """Bulk insert playlist track positions."""
    if not tracks:
        return 0
    rows = [(playlist_id, t["track_id"], t["position"]) for t in tracks]
    sql  = """
        INSERT INTO playlist_tracks (playlist_id, track_id, position)
        VALUES %s ON CONFLICT DO NOTHING
    """
    with db_cursor() as cur:
        execute_values(cur, sql, rows)
        return cur.rowcount


def log_run_start(source: str) -> int:
    """Log a scraper run start, return run ID."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO scraper_runs (source) VALUES (%s) RETURNING id",
            (source,)
        )
        return cur.fetchone()[0]


def log_run_finish(run_id: int, tracks: int, transitions: int,
                   playlists: int, errors: int, status: str, notes: str = ""):
    with db_cursor() as cur:
        cur.execute("""
            UPDATE scraper_runs SET
                finished_at       = NOW(),
                tracks_added      = %s,
                transitions_added = %s,
                playlists_added   = %s,
                errors            = %s,
                status            = %s,
                notes             = %s
            WHERE id = %s
        """, (tracks, transitions, playlists, errors, status, notes, run_id))
