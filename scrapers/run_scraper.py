"""
scrapers/run_scraper.py
========================
Main entrypoint for Cloud Run Jobs.
Reads SCRAPER_SOURCE env var (set by Terraform per job) and runs that source.

Usage locally (with venv active):
  SCRAPER_SOURCE=spotify GCP_PROJECT=your-project python scrapers/run_scraper.py
  SCRAPER_SOURCE=youtube GCP_PROJECT=your-project python scrapers/run_scraper.py

In Cloud Run the env vars are injected automatically.
"""

import os
import sys
import logging
import yaml

# Setup logging before anything else
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level    = getattr(logging, log_level, logging.INFO),
    format   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_scraper")

import utils.db as db
from utils.db import log_run_start, log_run_finish


def load_config(source: str) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "scraper_config.yaml")
    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    source_config  = full_config.get(source, {})
    global_config  = full_config.get("global", {})
    # Inject GCS bucket from env
    global_config["raw_backup_bucket"] = os.environ.get("GCS_BUCKET", "")
    return {**source_config, "_global": global_config}


def get_gcs_client(bucket_name: str):
    if not bucket_name:
        return None, None
    try:
        from google.cloud import storage
        client = storage.Client()
        return client, bucket_name
    except Exception as e:
        log.warning(f"GCS client unavailable: {e}")
        return None, None


def run(source: str):
    log.info(f"{'='*60}")
    log.info(f"Mixscope Scraper — source: {source.upper()}")
    log.info(f"{'='*60}")

    # Ensure DB tables exist
    try:
        db.create_tables()
    except Exception as e:
        log.error(f"DB init failed: {e}")
        sys.exit(1)

    config      = load_config(source)
    bucket_name = config["_global"].get("raw_backup_bucket", "")
    gcs_client, bucket = get_gcs_client(bucket_name)

    if not config.get("enabled", True):
        log.info(f"Source '{source}' is disabled in scraper_config.yaml — skipping.")
        return

    # Log run start
    run_id = log_run_start(source)
    stats  = {"tracks": 0, "transitions": 0, "playlists": 0, "errors": 0}
    status = "success"

    try:
        if source == "spotify":
            from spotify import SpotifyScraper
            scraper = SpotifyScraper(config, gcs_client, bucket)
            result  = scraper.run()
            stats["tracks"]     = result.get("tracks", 0)
            stats["playlists"]  = result.get("playlists", 0)
            stats["errors"]     = result.get("errors", 0)

        elif source == "youtube":
            from youtube import YouTubeScraper
            scraper = YouTubeScraper(config, gcs_client, bucket)
            result  = scraper.run()
            stats["tracks"]      = result.get("tracks", 0)
            stats["transitions"] = result.get("transitions", 0)
            stats["errors"]      = result.get("errors", 0)

        elif source == "mixcloud":
            from mixcloud import MixcloudScraper
            scraper = MixcloudScraper(config, gcs_client, bucket)
            result  = scraper.run()
            stats["tracks"]      = result.get("tracks", 0)
            stats["transitions"] = result.get("transitions", 0)
            stats["errors"]      = result.get("errors", 0)

        elif source == "onzerotracklists":
            from tracklists_1001 import Tracklists1001Scraper
            scraper = Tracklists1001Scraper(config, gcs_client, bucket)
            result  = scraper.run()
            stats["tracks"]      = result.get("tracks", 0)
            stats["transitions"] = result.get("transitions", 0)
            stats["errors"]      = result.get("errors", 0)

        elif source == "soundcloud":
            from soundcloud import SoundCloudScraper
            scraper = SoundCloudScraper(config, gcs_client, bucket)
            result  = scraper.run()
            stats["tracks"]      = result.get("tracks", 0)
            stats["transitions"] = result.get("transitions", 0)
            stats["errors"]      = result.get("errors", 0)

        elif source == "residentadvisor":
            from resident_advisor import ResidentAdvisorScraper
            scraper = ResidentAdvisorScraper(config, gcs_client, bucket)
            result  = scraper.run()
            stats["tracks"]      = result.get("tracks", 0)
            stats["transitions"] = result.get("transitions", 0)
            stats["errors"]      = result.get("errors", 0)

        else:
            log.error(f"Unknown source: '{source}'")
            sys.exit(1)

    except Exception as e:
        log.exception(f"Scraper failed with unhandled exception: {e}")
        status = "failed"
        stats["errors"] += 1

    finally:
        log_run_finish(
            run_id      = run_id,
            tracks      = stats["tracks"],
            transitions = stats["transitions"],
            playlists   = stats["playlists"],
            errors      = stats["errors"],
            status      = status,
            notes       = f"source={source}",
        )

    log.info(f"{'='*60}")
    log.info(f"Run complete — status: {status}")
    log.info(f"  Tracks added:      {stats['tracks']:,}")
    log.info(f"  Transitions added: {stats['transitions']:,}")
    log.info(f"  Playlists added:   {stats['playlists']:,}")
    log.info(f"  Errors:            {stats['errors']}")
    log.info(f"{'='*60}")

    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    source = os.environ.get("SCRAPER_SOURCE", "").lower()
    if not source:
        # Allow CLI override: python run_scraper.py spotify
        if len(sys.argv) > 1:
            source = sys.argv[1].lower()
        else:
            print("Usage: SCRAPER_SOURCE=spotify python run_scraper.py")
            print("       python run_scraper.py spotify")
            sys.exit(1)
    run(source)
