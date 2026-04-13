"""
scripts/train.py
================
Trains both Stage 1 (Random Forest) and Stage 2 (LightGBM Ranker).
All runs logged to MLflow — open `mlflow ui` to compare experiments.

Usage:
  python scripts/train.py                  # train both stages
  python scripts/train.py --stage rf       # random forest only
  python scripts/train.py --stage lgbm     # lightgbm only
  python scripts/train.py --stage lgbm --no-mlflow  # skip MLflow logging
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import config
from utils.metrics import evaluate_ranking, print_metrics_report

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

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


def load_split(name: str):
    path = f"{config.DATA_DIR}/{name}_pairs.npz"
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"].astype(int), list(map(tuple, data["pairs"]))


def train_random_forest(X_train, y_train, X_val, y_val, pairs_val, use_mlflow=True):
    print("\n" + "=" * 60)
    print("STAGE 1 — Random Forest Baseline")
    print("=" * 60)
    print(f"  Train: {X_train.shape}, positive rate: {y_train.mean():.3f}")
    print(f"  Val:   {X_val.shape}")

    model = RandomForestClassifier(**config.RF_PARAMS)

    print("\nTraining...")
    model.fit(X_train, y_train)

    # Validation metrics
    val_scores = model.predict_proba(X_val)[:, 1]
    val_auc    = roc_auc_score(y_val, val_scores)
    ranking_metrics = evaluate_ranking(
        query_ids  = [p[0] for p in pairs_val],
        scores     = val_scores,
        labels     = y_val,
        pair_index = pairs_val,
        k_values   = config.EVAL_K_VALUES,
    )

    print(f"\n  AUC-ROC: {val_auc:.4f}")
    print_metrics_report(ranking_metrics, stage="[RF / Val]")

    # Feature importances
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("  Feature importances (model-learned, after your config weights):")
    for fname, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"    {fname:<30} {imp:.4f}  {bar}")

    # Save
    os.makedirs("models/saved", exist_ok=True)
    model_path = "models/saved/random_forest.pkl"
    joblib.dump(model, model_path)
    print(f"\n  Saved → {model_path}")

    if use_mlflow:
        with mlflow.start_run(run_name="random_forest"):
            mlflow.log_params(config.RF_PARAMS)
            mlflow.log_params({f"weight_{k}": v for k, v in config.FEATURE_WEIGHTS.items()})
            mlflow.log_metric("val_auc_roc", val_auc)
            for k, v in ranking_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            mlflow.log_dict({f[0]: float(f[1]) for f in importances}, "feature_importances.json")
            mlflow.sklearn.log_model(model, "random_forest_model")
            print("  Logged to MLflow ✓")

    return model, ranking_metrics


def train_lightgbm(X_train, y_train, X_val, y_val, pairs_val, use_mlflow=True):
    try:
        import lightgbm as lgb
    except ImportError:
        print("⚠  LightGBM not installed. Run: pip install lightgbm")
        return None, {}

    print("\n" + "=" * 60)
    print("STAGE 2 — LightGBM LambdaRank")
    print("=" * 60)

    # LambdaRank needs group sizes (how many candidates per query)
    def make_groups(pairs):
        from collections import Counter
        c = Counter(p[0] for p in pairs)
        # Groups must be in same order as pairs
        seen = {}
        groups = []
        for p in pairs:
            qid = p[0]
            if qid not in seen:
                seen[qid] = c[qid]
                groups.append(c[qid])
        return groups

    train_groups = make_groups(
        list(np.load(f"{config.DATA_DIR}/train_pairs.npz", allow_pickle=True)["pairs"])
    )
    val_groups = make_groups(pairs_val)

    train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_data   = lgb.Dataset(X_val,   label=y_val,   group=val_groups, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    print("Training with LambdaRank (early stopping on NDCG@10)...")
    model = lgb.train(
        params           = config.LGBM_PARAMS,
        train_set        = train_data,
        valid_sets       = [val_data],
        valid_names      = ["val"],
        num_boost_round  = config.LGBM_PARAMS["n_estimators"],
        callbacks        = callbacks,
    )

    val_scores = model.predict(X_val)
    ranking_metrics = evaluate_ranking(
        query_ids  = [p[0] for p in pairs_val],
        scores     = val_scores,
        labels     = y_val,
        pair_index = pairs_val,
        k_values   = config.EVAL_K_VALUES,
    )
    print_metrics_report(ranking_metrics, stage="[LightGBM / Val]")

    # Feature importances
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True
    )
    print("  Feature importances (gain-based):")
    max_imp = max(v for _, v in importances) or 1
    for fname, imp in importances:
        bar = "█" * int(imp / max_imp * 40)
        print(f"    {fname:<30} {imp:8.1f}  {bar}")

    model_path = "models/saved/lightgbm_ranker.txt"
    model.save_model(model_path)
    print(f"\n  Saved → {model_path}")

    if use_mlflow:
        with mlflow.start_run(run_name="lightgbm_lambdarank"):
            mlflow.log_params(config.LGBM_PARAMS)
            mlflow.log_params({f"weight_{k}": v for k, v in config.FEATURE_WEIGHTS.items()})
            for k, v in ranking_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            mlflow.log_dict({f[0]: float(f[1]) for f in importances}, "feature_importances.json")
            mlflow.lightgbm.log_model(model, "lgbm_model")
            print("  Logged to MLflow ✓")

    return model, ranking_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      choices=["rf", "lgbm", "both"], default="both")
    parser.add_argument("--no-mlflow",  action="store_true")
    args = parser.parse_args()

    use_mlflow = not args.no_mlflow

    if use_mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{config.EXPERIMENTS_DIR}/mlflow.db")
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
        print(f"MLflow experiment: '{config.MLFLOW_EXPERIMENT}'")
        print(f"Run `mlflow ui --backend-store-uri sqlite:///{config.EXPERIMENTS_DIR}/mlflow.db` to view\n")

    print("Loading datasets...")
    X_train, y_train, pairs_train = load_split("train")
    X_val,   y_val,   pairs_val   = load_split("val")

    results = {}

    if args.stage in ("rf", "both"):
        _, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, pairs_val, use_mlflow)
        results["random_forest"] = rf_metrics

    if args.stage in ("lgbm", "both"):
        _, lgbm_metrics = train_lightgbm(X_train, y_train, X_val, y_val, pairs_val, use_mlflow)
        results["lightgbm"] = lgbm_metrics

    # Side-by-side comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON — Val Set")
        print("=" * 60)
        print(f"  {'Metric':<22}", end="")
        for name in results:
            print(f"  {name:>16}", end="")
        print()
        all_keys = sorted(set(k for m in results.values() for k in m))
        for key in all_keys:
            print(f"  {key:<22}", end="")
            for m in results.values():
                val = m.get(key, 0)
                print(f"  {val:>16.4f}", end="")
            print()

    print("\n✅ Training complete.")
    print("Next step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
