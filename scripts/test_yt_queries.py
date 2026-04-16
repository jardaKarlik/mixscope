"""
Test YouTube search queries for Mixscope scraper evaluation.
Fetches top 5 results per query (filtered to >20min), shows metadata
and tracklist detection signal. Flags likely false positives.
"""

import re
import sys
import time
from isodate import parse_duration
from google.cloud import secretmanager
from googleapiclient.discovery import build

GCP_PROJECT  = "mixsource"
SECRET_NAME  = "youtube_api_key"
YT_API       = "youtube"
YT_VERSION   = "v3"

QUERIES = [
    "freetekno DJ set tracklist",
    "raggatek mix tracklist",
    "teknival set tracklist",
    "acid tekno DJ set tracklist",
    "tribal tekno mix tracklist",
    "son de teuf DJ set",
    "free party tekno set tracklist",
    "hardtek DJ set tracklist",
    "raggajungle DJ set tracklist",
]

# Terms that suggest a false positive
FALSE_POSITIVE_SIGNALS = [
    # Nigerian artist Tekno
    r"\btekno miles\b", r"\btekno – ", r"\btekno - ",
    r"\bsign by tekno\b", r"\bnigerian\b", r"\bafrobeats?\b",
    r"\bafropop\b", r"\bnaija\b",
    # Mainstream techno (not free party / freetekno subculture)
    r"\bberghain\b", r"\bomar s\b", r"\bclockwork\b",
    r"\bnordic techno\b",
    # Clearly not a DJ set
    r"\blive band\b", r"\bofficial video\b", r"\bmusic video\b",
    r"\blyric video\b", r"\bkaraoke\b",
    # Wrong genre adjacent
    r"\bpsytrance\b", r"\bgoa\b", r"\bdrumstep\b",
]
FP_RE = re.compile("|".join(FALSE_POSITIVE_SIGNALS), re.IGNORECASE)


def get_api_key() -> str:
    # Read from temp file written by: gcloud secrets versions access latest \
    #   --secret="mixscope-youtube-api-key" --project="mixsource" > /tmp/yt_key.txt
    key_file = "/tmp/yt_key.txt"
    try:
        with open(key_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"Key file not found: {key_file}")


def has_tracklist(description: str) -> tuple[bool, str]:
    """
    Returns (found, reason) where reason explains what matched.
    Looks for:
      - Timestamps: 00:00 / 0:00 / 1:23:45
      - Numbered lines: 1. / 01. at line start
      - Track markers: track 1, #1 etc
    """
    if not description:
        return False, "no description"

    if re.search(r"^\s*\d{1,2}:\d{2}(:\d{2})?", description, re.MULTILINE):
        return True, "timestamps"
    if re.search(r"^\s*\d{1,2}\.\s+\S", description, re.MULTILINE):
        return True, "numbered lines"
    if re.search(r"^\s*#?\d+\s+[\w\"']", description, re.MULTILINE):
        return True, "track numbers"
    if re.search(r"\btracklist\b", description, re.IGNORECASE):
        return True, "keyword 'tracklist'"
    return False, "none"


def duration_minutes(iso: str) -> float:
    try:
        return parse_duration(iso).total_seconds() / 60
    except Exception:
        return 0.0


def flag_false_positives(title: str, channel: str) -> list[str]:
    flags = []
    combined = f"{title} {channel}"
    for pat in FALSE_POSITIVE_SIGNALS:
        if re.search(pat, combined, re.IGNORECASE):
            flags.append(pat.strip(r"\b").replace("\\b", "").replace("?", ""))
    return flags


def search_and_detail(yt, query: str, max_results: int = 20) -> list[dict]:
    """Search, then fetch details for videos, return up to 5 that pass >20min filter."""
    # search.list — 100 quota units
    search_resp = yt.search().list(
        part        = "id,snippet",
        q           = query,
        type        = "video",
        videoDuration = "long",   # >20 min — pre-filter
        maxResults  = max_results,
        order       = "relevance",
    ).execute()

    video_ids = [
        item["id"]["videoId"]
        for item in search_resp.get("items", [])
        if item.get("id", {}).get("videoId")
    ]
    if not video_ids:
        return []

    # videos.list — 1 quota unit per batch
    detail_resp = yt.videos().list(
        part = "snippet,contentDetails,statistics",
        id   = ",".join(video_ids),
    ).execute()

    results = []
    for item in detail_resp.get("items", []):
        snippet    = item.get("snippet", {})
        details    = item.get("contentDetails", {})
        statistics = item.get("statistics", {})

        dur_min    = duration_minutes(details.get("duration", "PT0S"))
        if dur_min < 20:
            continue

        title       = snippet.get("title", "")
        channel     = snippet.get("channelTitle", "")
        published   = snippet.get("publishedAt", "")[:10]
        view_count  = int(statistics.get("viewCount", 0))
        description = snippet.get("description", "")
        video_id    = item["id"]

        tl_found, tl_reason = has_tracklist(description)
        fp_flags            = flag_false_positives(title, channel)

        results.append({
            "id":          video_id,
            "title":       title,
            "channel":     channel,
            "dur_min":     dur_min,
            "views":       view_count,
            "published":   published,
            "tl_found":    tl_found,
            "tl_reason":   tl_reason,
            "fp_flags":    fp_flags,
        })
        if len(results) >= 5:
            break

    return results


def main():
    print("Fetching YouTube API key from Secret Manager...")
    try:
        api_key = get_api_key()
    except Exception as e:
        print(f"ERROR: Could not fetch API key: {e}")
        sys.exit(1)
    print("API key fetched OK.\n")

    yt = build(YT_API, YT_VERSION, developerKey=api_key)

    for i, query in enumerate(QUERIES, 1):
        print("=" * 70)
        print(f"QUERY {i}/{len(QUERIES)}: \"{query}\"")
        print("=" * 70)

        try:
            results = search_and_detail(yt, query)
        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(1)
            continue

        if not results:
            print("  No results (or all under 20 min)\n")
            continue

        for j, v in enumerate(results, 1):
            tl_icon = "✓" if v["tl_found"] else "✗"
            fp_note = f"  ⚠ FALSE POSITIVE? {v['fp_flags']}" if v["fp_flags"] else ""
            print(f"\n  [{j}] {v['title']}")
            print(f"      Channel  : {v['channel']}")
            print(f"      Duration : {v['dur_min']:.0f} min")
            print(f"      Views    : {v['views']:,}")
            print(f"      Uploaded : {v['published']}")
            print(f"      Tracklist: {tl_icon}  ({v['tl_reason']})")
            if fp_note:
                print(f"     {fp_note}")

        # Quota: each query costs ~101 units (100 search + 1 videos batch)
        # 9 queries = ~909 units, well within 10k/day
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print("Done. Estimated quota used: ~{} units.".format(len(QUERIES) * 101))


if __name__ == "__main__":
    main()
