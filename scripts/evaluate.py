"""
scripts/evaluate.py
===================
Final evaluation on the held-out TEST set.
Only run this when you're done tuning — test set is for final numbers only.

Usage:
  python scripts/evaluate.py              # evaluates best available model
  python scripts/evaluate.py --model rf   # force random forest
  python scripts/evaluate.py --model lgbm
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, classification_report

import config
from utils.metrics import evaluate_ranking, print_metrics_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "lgbm", "auto"], default="auto")
    args = parser.parse_args()

    print("=" * 60)
    print("MIXSCOPE — Final Test Set Evaluation")
    print("=" * 60)
    print("⚠  This is the held-out test set. Only run when done tuning.\n")

    # Load test data
    data = np.load(f"{config.DATA_DIR}/test_pairs.npz", allow_pickle=True)
    X_test, y_test, pairs_test = data["X"], data["y"].astype(int), list(map(tuple, data["pairs"]))
    print(f"Test set: {X_test.shape}, positive rate: {y_test.mean():.3f}")

    # Load model
    model_choice = args.model
    if model_choice == "auto":
        # Prefer LightGBM if available
        if os.path.exists("models/saved/lightgbm_ranker.txt"):
            model_choice = "lgbm"
        else:
            model_choice = "rf"

    if model_choice == "lgbm":
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file="models/saved/lightgbm_ranker.txt")
            scores = model.predict(X_test)
            model_name = "LightGBM LambdaRank"
        except Exception as e:
            print(f"Could not load LightGBM: {e}. Falling back to Random Forest.")
            model_choice = "rf"

    if model_choice == "rf":
        model = joblib.load("models/saved/random_forest.pkl")
        scores = model.predict_proba(X_test)[:, 1]
        model_name = "Random Forest"

    print(f"Model: {model_name}\n")

    # Classification metrics (for reference)
    preds = (scores >= config.RECOMMENDATION_THRESHOLD).astype(int)
    print("Classification Report (threshold={:.2f}):".format(config.RECOMMENDATION_THRESHOLD))
    print(classification_report(y_test, preds, target_names=["not mixed", "mixed"]))

    try:
        auc = roc_auc_score(y_test, scores)
        print(f"AUC-ROC: {auc:.4f}\n")
    except Exception:
        pass

    # Ranking metrics (what actually matters)
    metrics = evaluate_ranking(
        query_ids  = [p[0] for p in pairs_test],
        scores     = scores,
        labels     = y_test,
        pair_index = pairs_test,
        k_values   = config.EVAL_K_VALUES,
    )
    print_metrics_report(metrics, stage="[FINAL TEST SET]")

    # Guidance
    print("Interpreting results:")
    ndcg10 = metrics.get("ndcg@10", 0)
    hr10   = metrics.get("hit_rate@10", 0)
    p10    = metrics.get("precision@10", 0)

    if ndcg10 >= 0.7:
        print(f"  NDCG@10 = {ndcg10:.3f} — Excellent. Model ranks well.")
    elif ndcg10 >= 0.5:
        print(f"  NDCG@10 = {ndcg10:.3f} — Good. Try tuning weights in config.py.")
    else:
        print(f"  NDCG@10 = {ndcg10:.3f} — Needs work. Check data quality + weights.")

    if hr10 >= 0.8:
        print(f"  Hit Rate@10 = {hr10:.3f} — Correct track in top 10 most of the time.")
    else:
        print(f"  Hit Rate@10 = {hr10:.3f} — Model missing true matches. Consider lowering RECOMMENDATION_THRESHOLD.")

    print(f"\n  Feature weights used: {config.FEATURE_WEIGHTS}")
    print("\n  To change weights → edit config.py → re-run prepare_data.py + train.py → re-evaluate here.")


if __name__ == "__main__":
    main()
