"""
scripts/prepare_data.py
=======================
Builds the ML-ready dataset from raw corpus CSVs.

Input files (in data/):
  transitions.csv    — columns: track_a_id, track_b_id, set_id, set_date, source
  playlists.csv      — columns: playlist_id, track_id, position, source
  track_metadata.csv — columns: track_id, title, artist, bpm, camelot_key, genre,
                                 label, lastfm_tags, artist_sim_* (optional)

Output files (in data/):
  corpus_counts.pkl    — dict {(a,b): count}
  playlist_matrix.pkl  — dict {(a,b): weighted_count}
  train_pairs.npz      — X_train, y_train, pairs_train
  val_pairs.npz        — X_val,   y_val,   pairs_val
  test_pairs.npz       — X_test,  y_test,  pairs_test
  scaler.pkl           — fitted StandardScaler (apply to val/test only)
  feature_builder.pkl  — serialised FeatureBuilder
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from utils.features import FeatureBuilder
from utils.negatives import build_training_dataset

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs("models/saved", exist_ok=True)


def build_corpus_counts(transitions_df: pd.DataFrame) -> dict:
    """Count how often each (A→B) transition appears across all sets."""
    counts = defaultdict(int)
    for _, row in tqdm(transitions_df.iterrows(), total=len(transitions_df), desc="Counting transitions"):
        counts[(row["track_a_id"], row["track_b_id"])] += 1
    return dict(counts)


def build_playlist_matrix(playlists_df: pd.DataFrame, window: int = None) -> dict:
    """
    Sliding-window co-occurrence over playlists.
    Tracks within ±window positions count as co-occurring.
    Weight decays with distance: weight = 1 / distance.
    """
    window = window or config.PLAYLIST_WINDOW
    matrix = defaultdict(float)

    for playlist_id, group in tqdm(playlists_df.groupby("playlist_id"), desc="Building playlist matrix"):
        tracks = group.sort_values("position")["track_id"].tolist()
        for i, t_a in enumerate(tracks):
            for j in range(max(0, i - window), min(len(tracks), i + window + 1)):
                if i == j:
                    continue
                t_b    = tracks[j]
                dist   = abs(i - j)
                weight = 1.0 / dist
                matrix[(t_a, t_b)] += weight

    return dict(matrix)


def time_split(
    transitions_df: pd.DataFrame,
) -> tuple:
    """
    Split transitions by SET DATE to prevent future leakage.
    Returns train_df, val_df, test_df.
    """
    if "set_date" not in transitions_df.columns:
        print("⚠  No set_date column found — falling back to random 70/15/15 split.")
        n = len(transitions_df)
        idx = transitions_df.sample(frac=1, random_state=config.RANDOM_SEED).index
        return (
            transitions_df.loc[idx[:int(n*.70)]],
            transitions_df.loc[idx[int(n*.70):int(n*.85)]],
            transitions_df.loc[idx[int(n*.85):]],
        )

    transitions_df["set_date"] = pd.to_datetime(transitions_df["set_date"])
    train = transitions_df[transitions_df["set_date"] <  config.TRAIN_CUTOFF]
    val   = transitions_df[(transitions_df["set_date"] >= config.TRAIN_CUTOFF) &
                           (transitions_df["set_date"] <  config.VAL_CUTOFF)]
    test  = transitions_df[transitions_df["set_date"] >= config.VAL_CUTOFF]

    print(f"  Train: {len(train):>8,} transitions (before {config.TRAIN_CUTOFF})")
    print(f"  Val:   {len(val):>8,} transitions ({config.TRAIN_CUTOFF} – {config.VAL_CUTOFF})")
    print(f"  Test:  {len(test):>8,} transitions (after {config.VAL_CUTOFF})")
    return train, val, test


def main():
    print("=" * 60)
    print("MIXSCOPE — Data Preparation Pipeline")
    print("=" * 60)

    # ── Load raw data ─────────────────────────────────────────────
    print("\n[1/6] Loading raw CSVs...")
    transitions_df  = pd.read_csv(config.TRANSITIONS_CSV)
    playlists_df    = pd.read_csv(config.PLAYLISTS_CSV)
    track_meta_df   = pd.read_csv(config.TRACK_META_CSV)
    print(f"  Transitions:   {len(transitions_df):,}")
    print(f"  Playlist rows: {len(playlists_df):,}")
    print(f"  Tracks:        {len(track_meta_df):,}")

    # ── Build lookup tables ───────────────────────────────────────
    print("\n[2/6] Building corpus transition counts...")
    corpus_counts = build_corpus_counts(transitions_df)
    joblib.dump(corpus_counts, f"{config.DATA_DIR}/corpus_counts.pkl")
    print(f"  Unique (A→B) pairs: {len(corpus_counts):,}")

    print("\n[3/6] Building playlist co-occurrence matrix...")
    playlist_matrix = build_playlist_matrix(playlists_df, config.PLAYLIST_WINDOW)
    joblib.dump(playlist_matrix, f"{config.DATA_DIR}/playlist_matrix.pkl")
    print(f"  Unique co-occurrence pairs: {len(playlist_matrix):,}")

    # ── Time-based split ──────────────────────────────────────────
    print("\n[4/6] Splitting by set date (prevents future leakage)...")
    train_df, val_df, test_df = time_split(transitions_df)

    # ── Feature builder — fit normalisation on TRAIN ONLY ─────────
    print("\n[5/6] Building FeatureBuilder (fit on train only)...")
    train_corpus = build_corpus_counts(train_df)  # counts from training period only
    max_corpus   = max(train_corpus.values(), default=1)
    max_playlist = max(playlist_matrix.values(), default=1)

    fb = FeatureBuilder(
        corpus_counts    = train_corpus,
        playlist_matrix  = playlist_matrix,
        track_metadata   = track_meta_df,
        max_corpus_count = max_corpus,
        max_playlist_count = max_playlist,
    )
    joblib.dump(fb, "models/saved/feature_builder.pkl")

    # ── Build datasets ────────────────────────────────────────────
    print("\n[6/6] Building train / val / test datasets with hard negatives...")

    splits = [
        ("train", train_df),
        ("val",   val_df),
        ("test",  test_df),
    ]

    scaler     = StandardScaler()
    X_train_raw = None

    for split_name, split_df in splits:
        print(f"\n  → {split_name.upper()}")
        X, y, pairs = build_training_dataset(
            transitions_df = split_df,
            track_metadata = track_meta_df,
            feature_builder = fb,
            # Fewer negatives for val/test — just enough for evaluation
            ratio = config.NEGATIVE_RATIO if split_name == "train" else 5,
        )

        if split_name == "train":
            # ✅ Fit scaler ONLY on training data — never on val/test
            X_scaled = scaler.fit_transform(X)
            X_train_raw = X  # keep raw for sanity check
            joblib.dump(scaler, "models/saved/scaler.pkl")
            print(f"  StandardScaler fitted on training data.")
        else:
            # ✅ Only TRANSFORM val/test — no fitting
            X_scaled = scaler.transform(X)

        np.savez_compressed(
            f"{config.DATA_DIR}/{split_name}_pairs.npz",
            X=X_scaled, y=y, pairs=pairs,
        )
        print(f"  Saved {split_name}_pairs.npz — shape {X_scaled.shape}")

    print("\n✅ Data preparation complete.")
    print(f"   Feature weights applied: {config.FEATURE_WEIGHTS}")
    print("\nNext step: python scripts/train.py")


if __name__ == "__main__":
    main()
