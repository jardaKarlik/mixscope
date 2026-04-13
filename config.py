"""
config.py — YOUR CONTROL PANEL
================================
All model behaviour, feature weights, and hyperparameters live here.
Change these values, re-run train.py, compare results in MLflow.

You do NOT need to touch any other file to experiment with signal weighting.
"""

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
MODELS_DIR      = "models/saved"
EXPERIMENTS_DIR = "experiments"

# Auto-detects sample files if real corpus files don't exist yet.
# Once you have real data, drop transitions.csv / playlists.csv / track_metadata.csv
# into data/ and the sample files will be ignored automatically.
import os as _os
def _data(real, sample):
    real_path = f"{DATA_DIR}/{real}"
    return real_path if _os.path.exists(real_path) else f"{DATA_DIR}/{sample}"

TRANSITIONS_CSV = _data("transitions.csv",   "sample_transitions.csv")
PLAYLISTS_CSV   = _data("playlists.csv",     "sample_playlists.csv")
TRACK_META_CSV  = _data("track_metadata.csv","sample_track_metadata.csv")

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Time-based train/val/test split ─────────────────────────────────────────
# We split by SET DATE, not randomly — prevents future data leaking into training.
# DJ sets before TRAIN_CUTOFF → train
# Between TRAIN_CUTOFF and VAL_CUTOFF → validation
# After VAL_CUTOFF → test
TRAIN_CUTOFF = "2022-01-01"   # ~70% of corpus
VAL_CUTOFF   = "2023-01-01"   # ~15% of corpus
# remainder → test (~15%)

# ─── Negative sampling ───────────────────────────────────────────────────────
# How many negative (non-mixed) pairs to generate per positive pair.
# Higher ratio = more conservative model (fewer false positives).
# Lower ratio = more aggressive (more recommendations, lower precision).
# Start at 10, tune based on Precision@10 vs Hit Rate trade-off.
NEGATIVE_RATIO = 10

# Hard negative strategy: pick negatives that are "almost right" to force
# the model to learn fine distinctions.
# Options: "random" | "same_genre" | "bpm_adjacent" | "mixed"
HARD_NEGATIVE_STRATEGY = "mixed"  # recommended

# BPM window for bpm_adjacent hard negatives (±N BPM)
HARD_NEGATIVE_BPM_WINDOW = 15

# ─── FEATURE WEIGHTS ─────────────────────────────────────────────────────────
# This is the key tuning surface. Each weight multiplies that feature's
# contribution BEFORE the model sees it. Weight=1.0 means no change.
# Weight=2.0 doubles that signal's influence. Weight=0.0 disables it entirely.
#
# Start here when results don't match your musical intuition.
# Increase weight for signals you trust more for YOUR music taste.
# Decrease for signals that seem to mislead the model.
#
# After changing weights: re-run train.py → compare NDCG@10 in MLflow.
#
FEATURE_WEIGHTS = {
    # How often this exact pair appears in the DJ set corpus.
    # Strongest single signal for most genres. Start high.
    "corpus_transition_freq":   2.0,

    # How close the BPMs are (±0 = 1.0, ±15 BPM = ~0.0).
    # Important but NOT a hard rule — DJs pitch-shift constantly.
    # Lower this if your style crosses BPM boundaries a lot.
    "bpm_proximity":            1.0,

    # Camelot wheel distance (0 = same key, 6 = maximum distance).
    # Key rules are guidelines not laws. Set lower for harmonic mixing purists,
    # lower for more eclectic/experimental mixing.
    "key_harmony":              0.8,

    # How often both tracks appear in the same Spotify/YT playlist
    # within a sliding window of ±N tracks.
    # Very strong signal for genre/vibe clustering.
    "playlist_cooccurrence":    1.8,

    # Genre and label overlap, scene proximity via Last.fm tags.
    # Good general signal but can be too conservative.
    "genre_scene_proximity":    1.0,

    # Beat curve shape similarity (energy envelope match).
    # Your custom signal — unique to this system.
    # Captures whether the end of A flows into the start of B.
    "energy_curve_similarity":  1.2,

    # Same record label = often same sound.
    "label_overlap":            0.5,

    # Last.fm / MusicBrainz artist similarity score.
    "artist_similarity":        0.9,
}

# ─── Feature engineering settings ────────────────────────────────────────────
# Playlist co-occurrence window: how many tracks either side counts as "neighbor"
PLAYLIST_WINDOW = 5

# Log-transform highly skewed count features before scaling
LOG_TRANSFORM_FEATURES = [
    "corpus_transition_freq",
    "playlist_cooccurrence",
]

# ─── Model hyperparameters ────────────────────────────────────────────────────

# Stage 1: Random Forest (baseline, interpretable)
RF_PARAMS = {
    "n_estimators":    200,
    "max_depth":       12,
    "min_samples_split": 20,
    "min_samples_leaf":  10,
    "class_weight":    "balanced",   # handles our severe class imbalance
    "random_state":    RANDOM_SEED,
    "n_jobs":          -1,
}

# Stage 2: LightGBM Ranker (ranking-optimised, better for NDCG)
LGBM_PARAMS = {
    "objective":       "lambdarank",  # directly optimises NDCG
    "metric":          "ndcg",
    "ndcg_eval_at":    [5, 10],
    "num_leaves":      63,
    "learning_rate":   0.05,
    "n_estimators":    500,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "random_state":    RANDOM_SEED,
    "n_jobs":          -1,
    "verbose":         -1,
}

# ─── Evaluation settings ──────────────────────────────────────────────────────
EVAL_K_VALUES = [5, 10, 20]   # Precision@K, NDCG@K, Hit Rate@K

# Minimum score threshold to surface a recommendation
RECOMMENDATION_THRESHOLD = 0.35

# Max recommendations to return per query
MAX_RECOMMENDATIONS = 20

# ─── MLflow experiment name ───────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "mixscope-v1"
