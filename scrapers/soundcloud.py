"""
scrapers/soundcloud.py
=======================
Collects DJ set tracklists from SoundCloud.

Status: STUB — not yet implemented.
  SoundCloud's public API was severely restricted in 2019.
  Options to explore:
    A) SoundCloud API v2 (unofficial, no guaranteed SLA)
    B) Scrape track metadata from public set pages (HTML)
    C) Apply for official API access at soundcloud.com/you/apps

Requires: soundcloud-client-id in Secret Manager (once implemented)
"""

import logging

from base import BaseScraper

log = logging.getLogger(__name__)


class SoundCloudScraper(BaseScraper):
    """
    Stub scraper for SoundCloud.

    Config keys (scraper_config.yaml → soundcloud section):
        TBD
    """

    SOURCE = "soundcloud"

    def run(self) -> dict:
        log.info("SoundCloud scraper: not yet implemented.")
        return {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
