"""
scripts/predict.py
==================
Inference pipeline. Given a track, returns top K mix candidates
with full score breakdown per feature — matches what the UI shows.

Usage:
  python scripts/predict.py --track "Bicep - Glue" --top_k 10
  python scripts/predict.py --track_id "mbid_abc123" --top_k 5
  python scripts/predict.py --track "Bicep - Glue" --explain  # show score breakdown
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib

import config
from utils.features import FeatureBuilder, bpm_proximity_score, key_harmony_score

FEATURE_NAMES = [
    "corpus_transition_freq",
    "bpm_proximity",
    "key_harmony",
    "playlist_cooccurrence",
    "genre_scene_proximity",
    "energy_curve_similarity",
    "label_overlap",
    "artist_similarity",
]


def load_model():
    """Load best available model."""
    lgbm_path = "models/saved/lightgbm_ranker.txt"
    rf_path   = "models/saved/random_forest.pkl"

    if os.path.exists(lgbm_path):
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=lgbm_path)
            return model, "lgbm"
        except ImportError:
            pass

    if os.path.exists(rf_path):
        return joblib.load(rf_path), "rf"

    raise FileNotFoundError("No trained model found. Run: python scripts/train.py")


def find_track_id(query: str, track_metadata: pd.DataFrame) -> str:
    """Fuzzy match track name to track_id."""
    query_lower = query.lower()
    meta = track_metadata.copy()
    meta["_match"] = meta.apply(
        lambda r: (
            str(r.get("artist","")).lower() + " " +
            str(r.get("title","")).lower()
        ), axis=1
    )
    # Exact match first
    exact = meta[meta["_match"] == query_lower]
    if len(exact):
        return exact.iloc[0]["track_id"]

    # Partial match
    partial = meta[meta["_match"].str.contains(query_lower, na=False)]
    if len(partial):
        return partial.iloc[0]["track_id"]

    raise ValueError(f"Track not found: '{query}'. Try the exact 'Artist - Title' format.")


def predict(
    track_id: str,
    model,
    model_type: str,
    feature_builder: FeatureBuilder,
    track_metadata: pd.DataFrame,
    scaler,
    top_k: int = 10,
    explain: bool = False,
) -> list:
    """
    Returns list of dicts sorted by compatibility score descending.
    Each dict contains: track_id, artist, title, bpm, key, score, feature_scores.
    """
    all_ids = track_metadata["track_id"].tolist()
    candidates = [t for t in all_ids if t != track_id]

    # Build feature matrix for all candidates
    pairs = [(track_id, c) for c in candidates]
    X_raw = feature_builder.build_batch(pairs)

    # Apply scaler (transform only — scaler was fit on training data)
    X_scaled = scaler.transform(X_raw)

    # Score
    if model_type == "lgbm":
        scores = model.predict(X_scaled)
    else:
        scores = model.predict_proba(X_scaled)[:, 1]

    # Sort and take top K above threshold
    order    = np.argsort(scores)[::-1]
    results  = []
    meta_idx = track_metadata.set_index("track_id")

    for i in order:
        if scores[i] < config.RECOMMENDATION_THRESHOLD:
            break
        if len(results) >= top_k:
            break

        cid = candidates[i]
        row = meta_idx.loc[cid] if cid in meta_idx.index else {}

        result = {
            "rank":          len(results) + 1,
            "track_id":      cid,
            "artist":        row.get("artist", "Unknown"),
            "title":         row.get("title",  "Unknown"),
            "bpm":           row.get("bpm",    None),
            "camelot_key":   row.get("camelot_key", None),
            "genre":         row.get("genre",  None),
            "label":         row.get("label",  None),
            "compatibility": round(float(scores[i]) * 100, 1),
        }

        if explain:
            # Return raw (pre-scaled) feature values with their weights applied
            raw_features = X_raw[i]
            result["feature_scores"] = {
                name: {
                    "raw_weighted":  round(float(raw_features[j]), 4),
                    "config_weight": config.FEATURE_WEIGHTS[name],
                }
                for j, name in enumerate(FEATURE_NAMES)
            }

        results.append(result)

    return results


def print_results(results: list, explain: bool = False):
    print(f"\n{'─'*70}")
    print(f"  {'#':<3} {'Artist':<20} {'Title':<24} {'BPM':>4} {'Key':>4} {'Match':>6}")
    print(f"  {'─'*3} {'─'*20} {'─'*24} {'─'*4} {'─'*4} {'─'*6}")

    for r in results:
        print(f"  {r['rank']:<3} {str(r['artist']):<20} {str(r['title']):<24} "
              f"{str(r['bpm'] or '?'):>4} {str(r['camelot_key'] or '?'):>4} "
              f"{r['compatibility']:>5.1f}%")

        if explain and "feature_scores" in r:
            for fname, fdata in r["feature_scores"].items():
                bar = "▓" * int(fdata["raw_weighted"] * 20)
                w   = fdata["config_weight"]
                print(f"       {fname:<30} {fdata['raw_weighted']:.3f} (×{w}) {bar}")
            print()

    print(f"{'─'*70}\n")


def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--track",    type=str, help="'Artist - Title' string")
    group.add_argument("--track_id", type=str, help="Internal track_id")
    parser.add_argument("--top_k",   type=int, default=10)
    parser.add_argument("--explain", action="store_true", help="Show feature score breakdown")
    args = parser.parse_args()

    # Load artefacts
    print("Loading model and artefacts...")
    model, model_type = load_model()
    fb      = joblib.load("models/saved/feature_builder.pkl")
    scaler  = joblib.load("models/saved/scaler.pkl")
    meta_df = pd.read_csv(config.TRACK_META_CSV)

    # Resolve track ID
    if args.track_id:
        track_id = args.track_id
    else:
        track_id = find_track_id(args.track, meta_df)
        track_row = meta_df[meta_df["track_id"] == track_id].iloc[0]
        print(f"Matched: {track_row.get('artist','')} — {track_row.get('title','')} "
              f"[{track_row.get('bpm','')} BPM, {track_row.get('camelot_key','')}]")

    print(f"Scoring {len(meta_df)-1:,} candidates... (model: {model_type})")
    results = predict(
        track_id        = track_id,
        model           = model,
        model_type      = model_type,
        feature_builder = fb,
        track_metadata  = meta_df,
        scaler          = scaler,
        top_k           = args.top_k,
        explain         = args.explain,
    )

    print(f"\nTop {args.top_k} mix candidates:")
    print_results(results, explain=args.explain)

    if not results:
        print("No candidates above threshold. Try lowering RECOMMENDATION_THRESHOLD in config.py.")


if __name__ == "__main__":
    main()
