"""
Training script for competitive programming difficulty prediction models.
Trains both classification (Easy/Medium/Hard) and regression (0-10 score) models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    StackingRegressor,
    RandomForestClassifier
)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.feature_selection import VarianceThreshold

from preprocess import load_data, combine_text_columns
from features import add_meta_features

# Configuration
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Meta features used by both models
META_FEATURE_COLUMNS = [
    # Text statistics
    "text_len_log",
    "num_lines_est",
    "avg_word_len",
    "type_token_ratio",
    "token_count",
    "stopword_ratio",
    "punctuation_count",
    
    # Math and constraints
    "num_math_symbols",
    "num_constraints",
    "num_numerics",
    "max_number_log",
    "mean_number_log",
    "estimated_max_n",
    
    # Problem structure
    "num_examples",
    "num_sample_pairs",
    "has_big_o",
    "code_like_lines",
    "single_letter_vars",
    "question_mark",
    "has_directive_find",
    
    # Keywords (individual)
    "num_keywords",
    
    # Keyword groups
    "graph_kw_count",
    "dp_kw_count",
    "advanced_ds_kw_count",
    "math_kw_count",
    "string_kw_count",
    "geometry_kw_count",
    "greedy_kw_count",
]


def score_to_class(score: float) -> str:
    """Convert numeric difficulty score to categorical class."""
    if score <= 5.0:
        return "Easy"
    elif score <= 8.5:
        return "Medium"
    else:
        return "Hard"


def build_text_pipeline(max_features: int = 15000, n_components: int = 300) -> Pipeline:
    """
    Build TF-IDF + SVD pipeline for text feature extraction.
    
    Args:
        max_features: Maximum number of TF-IDF features
        n_components: Number of SVD components
        
    Returns:
        Sklearn Pipeline for text transformation
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=5,
            max_df=0.9,
            sublinear_tf=True,
            stop_words="english",
            strip_accents="unicode"
        )),
        ("svd", TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE,
            algorithm="randomized"
        ))
    ])


def prepare_feature_matrix(
    X_data: pd.DataFrame,
    text_pipeline: Pipeline,
    meta_cols: list,
    fit: bool = False
) -> np.ndarray:
    """
    Prepare combined feature matrix from text and meta features.
    
    Args:
        X_data: Input DataFrame with combined_text and meta features
        text_pipeline: Text transformation pipeline
        meta_cols: List of meta feature column names
        fit: Whether to fit the pipeline (True for training data)
        
    Returns:
        Combined feature matrix
    """
    # Transform text
    if fit:
        X_text = text_pipeline.fit_transform(X_data["combined_text"])
    else:
        X_text = text_pipeline.transform(X_data["combined_text"])
    
    # Ensure meta columns exist
    for col in meta_cols:
        if col not in X_data.columns:
            X_data[col] = 0
    
    # Extract meta features
    X_meta = X_data[meta_cols].values
    
    # Combine features
    X_combined = np.hstack([X_text, X_meta])
    
    return X_combined


def clean_and_scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    fit: bool = False,
    variance_threshold: float = 1e-8
):
    """
    Remove low-variance features and standardize.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix (optional)
        fit: Whether to fit transformers
        variance_threshold: Minimum variance for feature selection
        
    Returns:
        Transformed arrays and fitted transformers
    """
    if fit:
        # Fit variance filter
        vt = VarianceThreshold(variance_threshold)
        X_train = vt.fit_transform(X_train)
        
        # Fit scaler
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test = vt.transform(X_test)
            X_test = scaler.transform(X_test)
            return X_train, X_test, vt, scaler
        
        return X_train, vt, scaler
    else:
        raise ValueError("fit must be True for training data")


def train_classification_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series
) -> tuple:
    """
    Train and evaluate classification model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data.
        
    Returns:
        Trained model and predictions
    """
    print("\n" + "="*70)
    print("TRAINING CLASSIFICATION MODEL (Easy/Medium/Hard)")
    print("="*70)
    
    # Train LinearSVC
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        dual="auto"
    )
    
    print("\nFitting model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTraining Accuracy  : {train_acc:.4f}")
    print(f"Test Accuracy      : {test_acc:.4f}")
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test, labels=["Easy", "Medium", "Hard"])
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    print("="*70)
    
    return model, y_pred_test


def train_regression_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series
) -> tuple:
    """
    Train and evaluate regression model using stacking ensemble.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Trained model and predictions
    """
    print("\n" + "="*70)
    print("TRAINING REGRESSION MODEL (Difficulty Score 0-10)")
    print("="*70)
    
    # Define base models
    base_models = [
        ("ridge", Ridge(alpha=5.0, random_state=RANDOM_STATE)),
        ("hgb", HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ]
    
    # Stacking regressor
    model = StackingRegressor(
        estimators=base_models,
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 20.0]),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    print("\nFitting stacked ensemble...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n" + "-"*70)
    print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-"*70)
    print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
    print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
    print(f"{'R²':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
    print("-"*70)
    
    print("="*70)
    
    return model, y_pred_test


def save_model_artifacts(
    model,
    text_pipeline: Pipeline,
    scaler: StandardScaler,
    variance_filter: VarianceThreshold,
    meta_cols: list,
    filename: str
):
    """Save model and all preprocessing artifacts."""
    artifacts = {
        "model": model,
        "text_pipeline": text_pipeline,
        "scaler": scaler,
        "variance_filter": variance_filter,
        "meta_cols": meta_cols
    }
    
    filepath = ARTIFACT_DIR / filename
    joblib.dump(artifacts, filepath, compress=3)
    print(f"\n💾 Model saved to: {filepath}")


def save_predictions(
    y_true_class,
    y_pred_class,
    y_true_score,
    y_pred_score
):
    """Save test set predictions for analysis."""
    predictions_df = pd.DataFrame({
        "true_class": y_true_class,
        "pred_class": y_pred_class,
        "true_score": y_true_score,
        "pred_score": y_pred_score
    })
    
    filepath = ARTIFACT_DIR / "test_predictions.csv"
    predictions_df.to_csv(filepath, index=False)
    print(f"💾 Test predictions saved to: {filepath}")


def main():
    """Main training pipeline."""
    
    print("   COMPETITIVE PROGRAMMING DIFFICULTY PREDICTION MODEL TRAINING")
    
    
    # Load and preprocess data
    print("📂 Loading data...")
    try:
        df = load_data("problems_data_acm.jsonl")
    except FileNotFoundError:
        print("❌ Error: problems_data_acm.jsonl not found!")
        print("   Please ensure the data file is in the current directory.")
        return
    
    print(f"   Loaded {len(df)} problems")
    
    print("\n🔧 Preprocessing and feature engineering...")
    df = combine_text_columns(df)
    df = add_meta_features(df)
    
    # Create target variables
    df["problem_class"] = df["problem_score"].apply(score_to_class)
    
    # Check class distribution
    print("\n📊 Class Distribution:")
    print(df["problem_class"].value_counts().sort_index())
    print(f"\n   Score Range: {df['problem_score'].min():.2f} - {df['problem_score'].max():.2f}")
    print(f"   Score Mean : {df['problem_score'].mean():.2f} ± {df['problem_score'].std():.2f}")
    
    # Split data
    X = df
    y_class = df["problem_class"]
    y_score = df["problem_score"].astype(float)
    
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class
    )
    
    print(f"\n✂️  Data Split:")
    print(f"   Training  : {len(X_train)} samples")
    print(f"   Test      : {len(X_test)} samples")
    
    # Build text pipeline
    print("\n🏗️  Building feature extraction pipeline...")
    text_pipeline = build_text_pipeline()
    
    # Prepare feature matrices
    # Prepare feature matrices
    print("   Extacting features from training data...")
    X_train_combined = prepare_feature_matrix(
        X_train, text_pipeline, META_FEATURE_COLUMNS, fit=True
    )

    print("   Extracting features from test data...")
    X_test_combined = prepare_feature_matrix(
        X_test, text_pipeline, META_FEATURE_COLUMNS, fit=False  # ← Fixed here
    )
    
    print(f"   Feature matrix shape: {X_train_combined.shape}")
    
    # Clean and scale features
    print("\n🧹 Cleaning and scaling features...")
    X_train_scaled, X_test_scaled, vt, scaler = clean_and_scale_features(
        X_train_combined, X_test_combined, fit=True
    )
    
    print(f"   Final feature count: {X_train_scaled.shape[1]}")
    
    # Train classification model
    clf_model, y_class_pred = train_classification_model(
        X_train_scaled, y_class_train,
        X_test_scaled, y_class_test
    )
    
    save_model_artifacts(
        clf_model, text_pipeline, scaler, vt,
        META_FEATURE_COLUMNS,
        "classification_model.joblib"
    )
    
    # Train regression model
    reg_model, y_score_pred = train_regression_model(
        X_train_scaled, y_score_train,
        X_test_scaled, y_score_test
    )
    
    save_model_artifacts(
        reg_model, text_pipeline, scaler, vt,
        META_FEATURE_COLUMNS,
        "regression_model.joblib"
    )
    
    # Save predictions
    save_predictions(
        y_class_test, y_class_pred,
        y_score_test, y_score_pred
    )
    
    
    print("   TRAINING COMPLETE!")
    


if __name__ == "__main__":
    main()