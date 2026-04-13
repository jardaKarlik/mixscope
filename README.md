# Mixscope ML Pipeline

DJ track compatibility model trained on real DJ set transitions and playlist co-presence signals.

## Project Structure

```
mixscope_ml/
├── config.py                  # ← YOUR CONTROL PANEL: weights, thresholds, paths
├── data/
│   ├── sample_transitions.csv # Sample data to get started
│   └── sample_playlists.csv
├── utils/
│   ├── features.py            # Feature engineering (8 signals → feature vector)
│   ├── negatives.py           # Hard negative mining
│   └── metrics.py             # NDCG@K, Precision@K, Hit Rate
├── models/
│   ├── random_forest.py       # Stage 1: interpretable baseline
│   └── lightgbm_ranker.py     # Stage 2: ranking-optimised upgrade
├── scripts/
│   ├── prepare_data.py        # Build dataset from raw corpus
│   ├── train.py               # Main training entry point
│   ├── evaluate.py            # Full evaluation report
│   └── predict.py             # Inference: given track A → top K matches
└── experiments/               # MLflow experiment logs (auto-created)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Prepare dataset from your corpus CSVs
python scripts/prepare_data.py

# 2. Train
python scripts/train.py

# 3. Evaluate on test set
python scripts/evaluate.py

# 4. Get recommendations
python scripts/predict.py --track "Bicep - Glue" --top_k 10
```

## Tuning the model

Open `config.py` — all feature weights and model hyperparameters live there.
Change `FEATURE_WEIGHTS` to boost or suppress any signal. Re-run `train.py`.
Every run is logged to MLflow so you can compare experiments side by side.

```bash
mlflow ui  # → open http://localhost:5000
```
