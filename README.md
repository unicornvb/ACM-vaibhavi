
# CP Difficulty Predictor

**Predict competitive programming problem difficulty (Easy / Medium / Hard) and a numeric difficulty score (0–10) using machine learning.**

---

## Project overview

This repository implements a pipeline to predict the difficulty of competitive programming problems from their text (statement, input format and output format). It contains code for:

* data preprocessing (`src/preprocess.py`)
* feature extraction and meta-features (`src/features.py`)
* training classification and regression models (`train_models.py`)
* a Streamlit inference app (`predict_app.py`)
* evaluation & plotting utilities (`src/eval_and_plot.py`)

The models and preprocessing artifacts are saved to the `artifacts/` directory so the Streamlit app can load them for real-time inference.

---

## Repository structure

```
.
├── README.md
├── requirements.txt            # (recommended) Python deps
├── problems_data_acm.jsonl     # training dataset (JSONL input expected)
├── artifacts/                  # model + prediction outputs (generated)
│   ├── classification_model.joblib
│   ├── regression_model.joblib
│   ├── test_predictions.csv
│   ├── confusion_matrix.png
│   └── regression_scatter.png
├── train_models.py             # training pipeline (creates artifacts)
├── predict_app.py              # Streamlit app (inference & UI)
└── src/
    ├── preprocess.py
    ├── features.py
    └── eval_and_plot.py
```

---

## Quickstart

1. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate      # Windows (PowerShell: .venv\Scripts\Activate.ps1)
```

2. Install dependencies (example `requirements.txt` should include common packages):

```bash
pip install -r requirements.txt
```

Suggested packages (pin versions as needed): `numpy`, `pandas`, `scikit-learn`, `joblib`, `streamlit`, `matplotlib`, `seaborn`.

3. Place your training data file `problems_data_acm.jsonl` in the project root. The `src/preprocess.py` helper already supports `.jsonl` and CSV.

---

## Train models

To train both the classification (Easy/Medium/Hard) and regression (0–10 score) models and save artifacts:

```bash
python train_models.py
```

What this does (high level):

* loads `problems_data_acm.jsonl` (each line is a JSON problem object)
* combines description / input / output into `combined_text`
* extracts meta-features using `src/features.add_meta_features`
* builds TF-IDF + SVD text pipeline and appends meta features
* trains a LinearSVC classification model and a stacking regression ensemble
* saves both artifacts (joblib) to `artifacts/`
* writes `test_predictions.csv` for offline analysis

### Common training issues

* If `train_models.py` cannot find `problems_data_acm.jsonl`, place the file in the repository root or update the path in the script.
* If you hit `ValueError: unconverted data remains when parsing...` while parsing dates, check your data fields and formats. The included pipeline does not rely on date parsing by default.
* If you previously saw `AttributeError: module 'pandas' has no attribute 'read_jsonl'` — note that `pandas` has no built-in `read_jsonl` function in some versions; `src/preprocess.py` contains a `load_data()` helper that reads `.jsonl` safely.

---

## Run the Streamlit app (inference)

After training and artifact generation, launch the Streamlit app for real-time predictions:

```bash
streamlit run predict_app.py
```

The app expects the artifacts to be present in the `artifacts/` directory:

* `classification_model.joblib`
* `regression_model.joblib`

If models are missing, the app will prompt you to train them first.

---

## Evaluation & plots

Use `src/eval_and_plot.py` to generate evaluation visuals from a CSV of holdout predictions. By default the script expects `artifacts/holdout_predictions.csv` and will produce:

* `artifacts/confusion_matrix.png`
* `artifacts/regression_scatter.png`

You can also inspect `artifacts/test_predictions.csv` (created during training) for numeric analysis.

---

## Features (what `src/features.py` extracts)

The feature engine extracts a broad set of meta-features useful for problem difficulty prediction, including:

* text length, lines, token counts, average word length
* numeric counts (numbers, largest number), and derived logs
* presence of big-O notation, code-like lines, sample I/O counts
* counts of math/constraint phrases, punctuation and question marks
* detection of dozens of CP-specific keywords (graphs, dp, segment tree, trie, kmp, geometry, greedy, etc.) and grouped keyword counts
* derived features such as `estimated_max_n`, `num_keywords`, `type_token_ratio`

These features are saved under `meta_cols` and used alongside TF-IDF/SVD text vectors for modeling.

---

## Inference flow (what `predict_app.py` does)

1. Build `combined_text` from the three text fields.
2. Extract meta-features with `add_meta_features()`.
3. Load saved artifacts (text pipeline, scaler, variance filter, model).
4. Transform text and meta features into the model feature-space.
5. Return a predicted difficulty class and a clipped numeric score (0–10).
6. Streamlit UI displays detected features, metrics, and helpful hints.

---


