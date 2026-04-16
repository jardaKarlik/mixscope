"""
utils/features.py
=================
Builds the feature vector for each (track_a, track_b) pair.
Each signal is computed independently, then weighted by config.FEATURE_WEIGHTS,
then assembled into a single numpy array for model input.

Feature vector layout (9 dimensions):
  [0] corpus_transition_freq    — how often A→B in DJ sets
  [1] bpm_proximity             — BPM closeness score (0-1)
  [2] key_harmony               — Camelot wheel score (0-1)
  [3] playlist_cooccurrence     — playlist neighbor frequency
  [4] genre_scene_proximity     — genre/label/tag overlap
  [5] energy_curve_similarity   — beat curve shape match
  [6] label_overlap             — same record label flag
  [7] artist_similarity         — Last.fm artist similarity
  [8] source_quality_score      — best observed data quality for this pair (0-1)
                                   1.0=whitelist  0.5=search+tracklist  0.0=search+no-tl
                                   Built from source_quality_corpus at prepare_data time.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# ─── Camelot wheel: key → camelot position ────────────────────────────────────
# Standard Camelot notation used by DJs worldwide
CAMELOT = {
    "C major":  "8B",  "A minor":  "8A",
    "G major":  "9B",  "E minor":  "9A",
    "D major": "10B",  "B minor": "10A",
    "A major": "11B",  "F# minor":"11A",
    "E major": "12B",  "C# minor":"12A",
    "B major":  "1B",  "G# minor": "1A",
    "F# major": "2B",  "D# minor": "2A",
    "C# major": "3B",  "A# minor": "3A",
    "G# major": "4B",  "F minor":  "4A",
    "D# major": "5B",  "C minor":  "5A",
    "A# major": "6B",  "G minor":  "6A",
    "F major":  "7B",  "D minor":  "7A",
}

# Map camelot code → numeric position (number part)
def _camelot_number(code: str) -> Optional[int]:
    if not code or not isinstance(code, str):
        return None
    try:
        return int(code[:-1])
    except (ValueError, IndexError):
        return None

def _camelot_mode(code: str) -> Optional[str]:
    if not code or not isinstance(code, str):
        return None
    return code[-1] if code[-1] in ("A", "B") else None

def key_harmony_score(key_a: str, key_b: str) -> float:
    """
    Returns 0.0–1.0. 1.0 = same key. 0.0 = maximally incompatible.
    Compatible: same position (1.0), ±1 on wheel (0.85), A↔B same number (0.7),
    ±2 (0.5), ±3 (0.25), anything beyond (0.0).
    """
    if not key_a or not key_b:
        return 0.5  # unknown → neutral, don't penalise

    # Normalise via Camelot lookup if full key names passed
    if key_a in CAMELOT:
        key_a = CAMELOT[key_a]
    if key_b in CAMELOT:
        key_b = CAMELOT[key_b]

    n_a, m_a = _camelot_number(key_a), _camelot_mode(key_a)
    n_b, m_b = _camelot_number(key_b), _camelot_mode(key_b)

    if n_a is None or n_b is None:
        return 0.5

    # Circular distance on 12-point wheel
    dist = min(abs(n_a - n_b), 12 - abs(n_a - n_b))

    if dist == 0 and m_a == m_b:
        return 1.0   # identical
    if dist == 0 and m_a != m_b:
        return 0.7   # same number, A↔B (relative major/minor)
    if dist == 1 and m_a == m_b:
        return 0.85  # adjacent on wheel — very common DJ move
    if dist == 1:
        return 0.6
    if dist == 2:
        return 0.4
    if dist == 3:
        return 0.2
    return 0.0

def bpm_proximity_score(bpm_a: float, bpm_b: float) -> float:
    """
    Returns 0.0–1.0. 1.0 = identical BPM.
    Uses a smooth decay: within ±2 BPM = excellent, ±8 = good, ±15 = usable, beyond = poor.
    Accounts for half/double BPM (e.g., 70 and 140 are actually compatible).
    """
    if not bpm_a or not bpm_b or bpm_a <= 0 or bpm_b <= 0:
        return 0.5  # unknown → neutral

    def _score(a, b):
        diff = abs(a - b)
        if diff <= 2:   return 1.0
        if diff <= 5:   return 0.9
        if diff <= 8:   return 0.75
        if diff <= 12:  return 0.5
        if diff <= 15:  return 0.3
        if diff <= 20:  return 0.15
        return 0.0

    # Check direct and half/double time
    scores = [
        _score(bpm_a, bpm_b),
        _score(bpm_a * 2, bpm_b),
        _score(bpm_a, bpm_b * 2),
        _score(bpm_a / 2, bpm_b),
        _score(bpm_a, bpm_b / 2),
    ]
    return max(scores)


class FeatureBuilder:
    """
    Builds feature vectors for track pairs.
    Requires pre-loaded lookup tables (corpus counts, playlist matrix, metadata).
    All lookups are O(1) dict access — fast at inference time.
    """

    def __init__(
        self,
        corpus_counts: Dict,        # {(track_a_id, track_b_id): count}
        playlist_matrix: Dict,      # {(track_a_id, track_b_id): weighted_count}
        track_metadata: pd.DataFrame,  # track_id, bpm, key, genre, label, artist_sim_score
        max_corpus_count: float = None,
        max_playlist_count: float = None,
        source_quality_corpus: Dict = None,  # {(track_a_id, track_b_id): best_quality (1-3)}
    ):
        self.corpus   = corpus_counts
        self.playlist = playlist_matrix
        self.meta     = track_metadata.set_index("track_id") if "track_id" in track_metadata.columns else track_metadata
        self.weights  = config.FEATURE_WEIGHTS
        # Maps pair → best (lowest) source_quality seen across all training sets.
        # 1=whitelist, 2=search+tl, 3=search+no-tl. Default 2 if pair is unseen.
        self.sq_corpus = source_quality_corpus or {}

        # Normalisation denominators (fit on training data only — no leakage)
        self.max_corpus   = max_corpus_count   or max(corpus_counts.values(),  default=1)
        self.max_playlist = max_playlist_count or max(playlist_matrix.values(), default=1)

    def _get_meta(self, track_id: str, field: str, default=None):
        try:
            return self.meta.loc[track_id, field]
        except (KeyError, TypeError):
            return default

    def build(self, track_a_id: str, track_b_id: str) -> np.ndarray:
        """
        Returns weighted feature vector of shape (9,).
        Apply StandardScaler AFTER building the full dataset — not here.
        """
        # ── Raw feature values ─────────────────────────────────────────────────

        # [0] Corpus transition frequency (log-normalised, bidirectional)
        count_ab = self.corpus.get((track_a_id, track_b_id), 0)
        count_ba = self.corpus.get((track_b_id, track_a_id), 0)
        corpus_raw = (count_ab + 0.3 * count_ba)  # forward transition weighted higher
        corpus_feat = np.log1p(corpus_raw) / np.log1p(self.max_corpus)

        # [1] BPM proximity
        bpm_a = self._get_meta(track_a_id, "bpm", 0)
        bpm_b = self._get_meta(track_b_id, "bpm", 0)
        bpm_feat = bpm_proximity_score(float(bpm_a or 0), float(bpm_b or 0))

        # [2] Key harmony
        key_a = self._get_meta(track_a_id, "camelot_key", "")
        key_b = self._get_meta(track_b_id, "camelot_key", "")
        key_feat = key_harmony_score(str(key_a or ""), str(key_b or ""))

        # [3] Playlist co-occurrence (log-normalised)
        pl_raw  = self.playlist.get((track_a_id, track_b_id), 0)
        pl_raw += self.playlist.get((track_b_id, track_a_id), 0) * 0.5  # symmetric but directional
        pl_feat = np.log1p(pl_raw) / np.log1p(self.max_playlist)

        # [4] Genre / scene proximity
        genre_a  = self._get_meta(track_a_id, "genre", "")
        genre_b  = self._get_meta(track_b_id, "genre", "")
        label_a  = self._get_meta(track_a_id, "label", "")
        label_b  = self._get_meta(track_b_id, "label", "")
        tags_a   = set(str(self._get_meta(track_a_id, "lastfm_tags", "") or "").lower().split(","))
        tags_b   = set(str(self._get_meta(track_b_id, "lastfm_tags", "") or "").lower().split(","))
        genre_score = 1.0 if genre_a and genre_a == genre_b else 0.3
        label_score = 1.0 if label_a and label_a == label_b else 0.0
        tag_score   = len(tags_a & tags_b) / max(len(tags_a | tags_b), 1) if tags_a and tags_b else 0.3
        genre_feat  = (genre_score * 0.4 + label_score * 0.2 + tag_score * 0.4)

        # [5] Energy curve similarity (0-1, pre-computed and stored in metadata)
        # This column is populated by the audio analysis pipeline (beat curve extractor)
        # Falls back to 0.5 (neutral) if not available yet
        ecurve_feat = float(self._get_meta(track_a_id, f"energy_sim_{track_b_id}", 0.5) or 0.5)
        # Alternative: use a stored similarity matrix if available
        # ecurve_feat = energy_sim_matrix.get((track_a_id, track_b_id), 0.5)

        # [6] Label overlap (binary, already computed above — separate for weighting)
        label_feat = label_score

        # [7] Artist similarity (pre-computed from Last.fm/MusicBrainz, stored in metadata)
        artist_sim = self._get_meta(track_a_id, f"artist_sim_{track_b_id}", None)
        if artist_sim is None:
            # Fallback: check if same artist
            artist_a = self._get_meta(track_a_id, "artist", "")
            artist_b = self._get_meta(track_b_id, "artist", "")
            artist_sim = 0.9 if artist_a and artist_a == artist_b else 0.3
        artist_feat = float(artist_sim)

        # [8] Source quality score for this pair
        # Best (minimum) source_quality seen for (A→B) in training corpus.
        # Normalized: quality 1 → 1.0 (high trust), 2 → 0.5, 3 → 0.0 (low trust)
        # Default 2 (0.5) for unseen pairs — neutral, neither rewarded nor penalised.
        best_quality = min(
            self.sq_corpus.get((track_a_id, track_b_id), 2),
            self.sq_corpus.get((track_b_id, track_a_id), 2),
        )
        sq_feat = (3 - best_quality) / 2.0  # maps 1→1.0, 2→0.5, 3→0.0

        # ── Raw feature vector ─────────────────────────────────────────────────
        raw = np.array([
            corpus_feat,    # [0]
            bpm_feat,       # [1]
            key_feat,       # [2]
            pl_feat,        # [3]
            genre_feat,     # [4]
            ecurve_feat,    # [5]
            label_feat,     # [6]
            artist_feat,    # [7]
            sq_feat,        # [8]
        ], dtype=np.float32)

        # ── Apply config weights ───────────────────────────────────────────────
        weight_vec = np.array([
            self.weights["corpus_transition_freq"],
            self.weights["bpm_proximity"],
            self.weights["key_harmony"],
            self.weights["playlist_cooccurrence"],
            self.weights["genre_scene_proximity"],
            self.weights["energy_curve_similarity"],
            self.weights["label_overlap"],
            self.weights["artist_similarity"],
            self.weights.get("source_quality_score", 1.0),
        ], dtype=np.float32)

        return raw * weight_vec

    def build_batch(self, pairs: list) -> np.ndarray:
        """Build features for a list of (track_a_id, track_b_id) tuples."""
        return np.array([self.build(a, b) for a, b in pairs])

    @property
    def feature_names(self):
        return [
            "corpus_transition_freq",
            "bpm_proximity",
            "key_harmony",
            "playlist_cooccurrence",
            "genre_scene_proximity",
            "energy_curve_similarity",
            "label_overlap",
            "artist_similarity",
            "source_quality_score",
        ]
