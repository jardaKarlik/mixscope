"""
scrapers/resident_advisor.py
=============================
Collects DJ set tracklists from Resident Advisor (ra.co).

Status: STUB — not yet implemented.
  RA publishes podcast tracklists (RA Exchange, RA Podcast series).
  Options to explore:
    A) RA has a GraphQL API used by their web app — reverse-engineer endpoints
    B) Scrape tracklist pages from ra.co/podcast and ra.co/exchange
    C) Contact RA for data partnership

Flow (planned):
  1. Browse podcast listing pages by genre/series
  2. For each podcast page: scrape structured tracklist
  3. Write sets + transitions to DB

Requires: no API key for public pages (once implemented)
"""

import logging

from base import BaseScraper

log = logging.getLogger(__name__)


class ResidentAdvisorScraper(BaseScraper):
    """
    Stub scraper for Resident Advisor.

    Config keys (scraper_config.yaml → residentadvisor section):
        TBD
    """

    SOURCE = "residentadvisor"

    def run(self) -> dict:
        log.info("Resident Advisor scraper: not yet implemented.")
        return {"sets": 0, "tracks": 0, "transitions": 0, "errors": 0}
