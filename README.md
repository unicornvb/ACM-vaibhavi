# Competitive Programming Difficulty Predictor 

An end-to-end machine learning system that predicts the difficulty of competitive programming problems using natural language processing and ensemble learning techniques.

Here is the video link- https://drive.google.com/drive/folders/1fTZT2YSWa2meekuKpVdw8q5u14Yluvw9?usp=sharing

**System Outputs:**
- **Difficulty Class:** Easy / Medium / Hard
- **Numeric Difficulty Score:** 0–10

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Feature Engineering](#-feature-engineering)
- [Models](#-models)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Alternative Approach: SBERT + TF-IDF](#-alternative-approach-sbert--tf-idf)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#%EF%B8%8F-usage)
- [Web Application](#-web-application)
- [Future Work](#-future-work)


---

## 🌟 Project Overview

This project implements a machine learning system to automatically estimate the difficulty of competitive programming problems based on their problem statements. Instead of relying on manually assigned difficulty tags, the system learns patterns from textual descriptions, constraints, and algorithmic cues.

**Key Features:**
- TF-IDF based text representations
- Domain-specific feature engineering
- Supervised classification and regression models
- Streamlit-based web application for real-time predictions

---


## 📊 Dataset

- **Dataset File:** `problems_data_acm.jsonl`
- **Format:** JSON Lines (JSONL)
- **Domain:** ACM-style competitive programming problems

### Data Fields

Each problem entry contains:

```json
{
  "description": "Main problem statement",
  "input_description": "Input format and constraints",
  "output_description": "Expected output format",
  "problem_score": 6.5,
  "problem_class": "Medium",
  "url": "https://example.com/problem-link"
}
```

| Field                | Description                            |
|----------------------|----------------------------------------|
| `description`        | Core problem statement                 |
| `input_description`  | Input format and constraints           |
| `output_description` | Output format                          |
| `problem_score`      | Numeric difficulty score (0–10)        |
| `problem_class`      | Difficulty label (Easy / Medium / Hard)|
| `url`                | Original problem URL                   |

> **Note:** `problem_class` is derived from `problem_score` using predefined thresholds during training.

---

## 🧠 Methodology

The project follows a structured machine learning pipeline:

1. Load and preprocess problem text
2. Combine problem description, input, and output into a unified text field
3. Extract textual and structural features
4. Train classification and regression models
5. Evaluate performance on a held-out test set
6. Deploy trained models via a web interface

---

## 🧩 Feature Engineering

### Text Features

- **TF-IDF vectorization** using unigrams and bigrams
- Maximum vocabulary size: 15,000
- Dimensionality reduction using Truncated SVD (300 components)
- Stopword removal

### Meta Features

Handcrafted features extracted from problem statements:

- Text length and structure
- Numeric constraints and estimated maximum input size
- Presence of examples and sample I/O
- Detection of Big-O notation
- Code-like patterns
- Algorithm-specific keywords

**Keyword Groups:**
- Graph algorithms
- Dynamic programming
- Advanced data structures
- String algorithms
- Mathematics
- Geometry
- Greedy techniques

These features help capture problem complexity beyond raw text.

---

## 🤖 Models

### Classification Model

- **Algorithm:** Linear Support Vector Classifier (LinearSVC)
- **Class Weighting:** Balanced
- **Scaling:** StandardScaler
- **Feature Selection:** VarianceThreshold

### Regression Model

- **Algorithm:** Stacking Regressor
- **Base Models:** Ridge Regression, Histogram Gradient Boosting
- **Final Estimator:** RidgeCV

Both models use the same combined feature space.

---

## 📈 Evaluation

### Classification Results (Test Set)

**Overall Accuracy:** 57.23%

| Class  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Easy   | 0.63      | 0.59   | 0.61     |
| Medium | 0.59      | 0.60   | 0.60     |
| Hard   | 0.05      | 0.07   | 0.06     |

> The model performs reliably for Easy and Medium problems but struggles with Hard problems due to class imbalance and label subjectivity.

### Regression Results (Test Set)

| Metric | Value |
|--------|-------|
| MAE    | 1.65  |
| RMSE   | 1.99  |
| R²     | 0.18  |

Predicted difficulty scores are typically within ±1–2 points of the true score.

---

## 📁 Project Structure

```
ACM_PROJECT_3/
│
├── artifacts/                     # Model files and evaluation outputs
│   ├── classification_model.joblib
│   ├── regression_model.joblib
│   ├── test_predictions.csv
│   ├── confusion_matrix.png
│   └── regression_scatter.png
│
├── __pycache__/                   # Auto-generated cache files
│
├── eval_and_plot.py               # Evaluation and visualization
├── features.py                    # Feature engineering
├── preprocess.py                  # Data preprocessing
├── train_models.py                # Model training pipeline
├── predict_app.py                 # Streamlit web application
│
├── problems_data_acm.jsonl        # Dataset
├── acm_doc.pdf                    # Project documentation
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🔁 Alternative Approach: SBERT + TF-IDF

During development, I experimented with combining **SBERT (Sentence-BERT)** embeddings with TF-IDF features. My initial hypothesis was that deep semantic representations would improve difficulty prediction.

However, this approach led to lower classification accuracy (below 50%). After analysis, I found that competitive programming difficulty depends heavily on **rare algorithmic keywords** such as "segment tree," "bitmask," and "flow." TF-IDF assigns higher importance to such rare but informative terms, whereas SBERT smooths them into dense semantic representations.

**Conclusion:** TF-IDF preserved sharp difficulty cues more effectively than SBERT. Based on empirical results, I chose **TF-IDF + meta-features** as the final approach.

---

## ⚙️ Installation

### Requirements

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train Models

```bash
python train_models.py
```

### Generate Evaluation Plots

```bash
python eval_and_plot.py
```

### Run Web Application

```bash
streamlit run predict_app.py
```

---

## 🌐 Web Application

The Streamlit interface allows users to:

- Enter a problem statement
- Add input and output formats
- Predict difficulty class and score
- View extracted features and algorithm hints
- Test predictions using example problems

The interface is designed to be simple and interpretable.

---

## 🔮 Future Work

- Improve class balance using sampling techniques
- Explore ordinal regression methods
- Introduce transformer-based embeddings with task-specific fine-tuning
- Add explainability tools such as SHAP or LIME
- Extend support to other competitive programming platforms

---

