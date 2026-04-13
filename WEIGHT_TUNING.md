# Weight Tuning Guide
## How to shape the model to your musical taste

The model has 8 signals. Each has a weight in `config.py → FEATURE_WEIGHTS`.
This file explains what each signal captures, when to increase or decrease it,
and what to watch for in the MLflow results after each change.

---

## The 8 Signals

### 1. `corpus_transition_freq` (default: 2.0)
**What it is:** How many times this exact A→B transition appears across all scraped DJ sets.

**What high weight means:** The model heavily favours pairs that professionals actually played together. Conservative, proven choices.

**When to increase:** You want recommendations that feel "validated" by the DJ community. Great for mainstream genres where the corpus is dense.

**When to decrease:** Your music taste is niche or experimental — fewer sets in the corpus means this signal is sparse and less reliable. If you're playing obscure vinyl-only techno, the corpus may not have enough data here.

**Red flag:** If NDCG@10 is high but recommendations feel "safe/boring" → lower this.

---

### 2. `bpm_proximity` (default: 1.0)
**What it is:** How close the BPMs are. ±0 = score 1.0. ±15 BPM = score 0.3. Also checks half/double time (70↔140).

**When to increase:** You mix precise beatmatch-style. BPM accuracy matters.

**When to decrease:** You scratch, pitch-shift aggressively, or use breaks/transitions that don't need tight BPM matching.

**Note:** The model already accounts for half/double time automatically.

---

### 3. `key_harmony` (default: 0.8)
**What it is:** Camelot wheel compatibility. Same key = 1.0, adjacent = 0.85, 2 steps = 0.4, beyond = 0.

**When to increase:** You practice harmonic mixing. Key clashes ruin your sets.

**When to decrease:** You use EQ to mask key clashes, or you play genres where key doesn't matter (hard techno, noise, industrial).

**Important insight:** The corpus signal will override key harmony if professionals regularly mix "wrong key" combinations. That's the model working correctly — sometimes the right vibe beats the theory.

---

### 4. `playlist_cooccurrence` (default: 1.8)
**What it is:** How often A and B appear within 5 tracks of each other across Spotify, YouTube, and other playlists. Weighted by proximity — neighbors are stronger than 5-apart.

**When to increase:** You curate from playlists a lot. You trust editorial taste.

**When to decrease:** Your music isn't well-represented in mainstream playlists. If you play very underground/unreleased stuff, this signal may be noise.

**Key advantage:** This is the signal that finds "feels like" matches that corpus transitions might miss — tracks that inhabit the same sonic world without being explicitly mixed together.

---

### 5. `genre_scene_proximity` (default: 1.0)
**What it is:** Combination of genre tag match, record label overlap, and Last.fm tag Jaccard similarity.

**When to increase:** You stay within a scene. Berlin techno stays Berlin techno.

**When to decrease:** You're an eclectic selector. You cross genres intentionally. This signal will actively work against genre-crossing recommendations if weighted too high.

---

### 6. `energy_curve_similarity` (default: 1.2)
**What it is:** Similarity of the beat curve energy envelope shape — does the end of track A match the energy profile of the start of track B.

**This is your unique signal.** No other DJ tool uses this.

**When to increase:** You care deeply about energy flow. You hate jarring energy drops.

**When to decrease:** You deliberately use energy contrast as a mixing technique (big drop, then quiet intro for tension).

**Note:** This signal requires the audio analysis pipeline to populate the metadata. Falls back to 0.5 (neutral) if not computed yet.

---

### 7. `label_overlap` (default: 0.5)
**What it is:** Binary — 1.0 if same record label, 0 otherwise.

**When to increase:** You buy from specific labels whose aesthetic is consistent (e.g., all Hessle Audio, all Ostgut Ton).

**When to decrease:** You mix across labels freely. This is a weak signal for most DJs.

---

### 8. `artist_similarity` (default: 0.9)
**What it is:** Last.fm/MusicBrainz artist similarity score, or 0.9 if same artist.

**When to increase:** You follow artist networks closely.

**When to decrease:** You rarely double up on artists in a set and prefer variety.

---

## Workflow for tuning

```
1. Run baseline: python scripts/train.py && python scripts/evaluate.py
   Note your NDCG@10 and Hit Rate@10.

2. Change ONE weight in config.py (increase or decrease by 0.3–0.5)

3. python scripts/prepare_data.py  # rebuild features with new weights
   python scripts/train.py
   python scripts/evaluate.py

4. Compare in MLflow: mlflow ui
   Did NDCG@10 go up? Did Hit Rate@10 improve?

5. More importantly: python scripts/predict.py --track "Your Track" --explain
   Do the recommendations feel right to YOUR ears?
   A model that matches your taste > a model with high NDCG on a corpus you didn't curate.

6. Keep a notes file. After each run write:
   weights → [corpus:2.0, bpm:1.0 ...] → NDCG@10: 0.72 → felt: "too conservative"
```

---

## Signs of a good model (for your taste)

- Recommendations you'd actually play in a real set
- Some surprises that turn out to be right
- Corpus frequency and playlist co-presence are the dominant features in the importance chart
- NDCG@10 > 0.65

## Signs the model needs tuning

- Everything it recommends is "obvious" (corpus_freq too high)
- Everything is in the same BPM/key (bpm or key weights too high)
- Recommendations feel genre-locked (genre_scene too high)
- Results feel random (negative ratio too low, or corpus too sparse)
