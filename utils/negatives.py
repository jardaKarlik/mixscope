"""
utils/negatives.py
==================
Hard negative mining for training data construction.

The core problem: our positive pairs (A→B actually mixed) are ~1.2M.
Possible negative pairs (A→B never mixed) are in the billions.
We need smart negatives — tracks that LOOK mixable but weren't chosen —
so the model learns real distinctions, not just "jazz vs techno".

Three strategies (configured in config.HARD_NEGATIVE_STRATEGY):
  "random"       — fully random negatives. Fast, weak.
  "same_genre"   — same genre, different BPM range. Better.
  "bpm_adjacent" — within ±15 BPM of A, never mixed. Hardest.
  "mixed"        — blend of all three. Recommended.
"""

import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Set
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def build_negative_pairs(
    positive_pairs: List[Tuple[str, str]],
    track_metadata: pd.DataFrame,
    positive_set: Set[Tuple[str, str]],
    ratio: int = None,
    strategy: str = None,
) -> List[Tuple[str, str]]:
    """
    Given positive pairs and track metadata, generate hard negative pairs.

    Args:
        positive_pairs:  list of (track_a_id, track_b_id) that ARE mixed
        track_metadata:  DataFrame with track_id, bpm, genre, label columns
        positive_set:    set of all known positive pairs (for exclusion)
        ratio:           negatives per positive (default: config.NEGATIVE_RATIO)
        strategy:        "random"|"same_genre"|"bpm_adjacent"|"mixed"

    Returns:
        List of (track_a_id, track_b_id) negative pairs
    """
    ratio    = ratio    or config.NEGATIVE_RATIO
    strategy = strategy or config.HARD_NEGATIVE_STRATEGY

    meta = track_metadata.set_index("track_id") if "track_id" in track_metadata.columns else track_metadata
    all_ids = list(meta.index)
    id_set  = set(all_ids)

    # Build genre and BPM lookup for fast filtering
    genre_index = {}  # genre → list of track_ids
    for tid in all_ids:
        g = str(meta.loc[tid, "genre"] if "genre" in meta.columns else "unknown")
        genre_index.setdefault(g, []).append(tid)

    bpm_arr = {}  # track_id → bpm float
    if "bpm" in meta.columns:
        for tid in all_ids:
            try:
                bpm_arr[tid] = float(meta.loc[tid, "bpm"] or 0)
            except (ValueError, TypeError):
                bpm_arr[tid] = 0.0

    negatives = []
    n_needed  = len(positive_pairs) * ratio

    # Distribute budget across strategies
    if strategy == "mixed":
        budgets = {
            "bpm_adjacent": int(n_needed * 0.5),
            "same_genre":   int(n_needed * 0.3),
            "random":       int(n_needed * 0.2),
        }
    else:
        budgets = {strategy: n_needed}

    for strat, budget in budgets.items():
        generated = 0
        attempts  = 0
        max_attempts = budget * 20  # avoid infinite loop on small datasets

        while generated < budget and attempts < max_attempts:
            attempts += 1

            # Pick a random anchor track from a positive pair
            pair_a, _ = random.choice(positive_pairs)

            if strat == "random":
                candidate = random.choice(all_ids)

            elif strat == "same_genre":
                genre = str(meta.loc[pair_a, "genre"]) if "genre" in meta.columns else "unknown"
                pool  = genre_index.get(genre, all_ids)
                if len(pool) < 2:
                    candidate = random.choice(all_ids)
                else:
                    candidate = random.choice(pool)

            elif strat == "bpm_adjacent":
                bpm_a = bpm_arr.get(pair_a, 0)
                if bpm_a <= 0:
                    candidate = random.choice(all_ids)
                else:
                    window = config.HARD_NEGATIVE_BPM_WINDOW
                    pool   = [
                        tid for tid in all_ids
                        if abs(bpm_arr.get(tid, 0) - bpm_a) <= window
                        and bpm_arr.get(tid, 0) > 0
                    ]
                    if not pool:
                        candidate = random.choice(all_ids)
                    else:
                        candidate = random.choice(pool)
            else:
                candidate = random.choice(all_ids)

            # Reject if: same track, already a positive pair, or already generated
            if (
                candidate == pair_a
                or (pair_a, candidate) in positive_set
                or (candidate, pair_a) in positive_set
            ):
                continue

            negatives.append((pair_a, candidate))
            generated += 1

    return negatives


def build_training_dataset(
    transitions_df: pd.DataFrame,
    track_metadata: pd.DataFrame,
    feature_builder,
    ratio: int = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Full pipeline: transitions → positive pairs → hard negatives → feature matrix.

    Returns:
        X:     feature matrix (N, 8)
        y:     labels (N,) — 1 for positive, 0 for negative
        pairs: list of (track_a_id, track_b_id) — for debugging
    """
    print("Building positive pairs from transitions...")
    positive_pairs = list(zip(transitions_df["track_a_id"], transitions_df["track_b_id"]))
    positive_set   = set(positive_pairs) | {(b, a) for a, b in positive_pairs}
    print(f"  → {len(positive_pairs):,} positive pairs")

    print(f"Mining hard negatives (strategy={config.HARD_NEGATIVE_STRATEGY}, ratio={ratio or config.NEGATIVE_RATIO})...")
    negative_pairs = build_negative_pairs(
        positive_pairs, track_metadata, positive_set, ratio=ratio
    )
    print(f"  → {len(negative_pairs):,} negative pairs")

    all_pairs  = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    print("Building feature vectors...")
    X = feature_builder.build_batch(all_pairs)
    y = np.array(all_labels, dtype=np.int32)

    print(f"  → Dataset shape: {X.shape}, positive rate: {y.mean():.3f}")
    return X, y, all_pairs
