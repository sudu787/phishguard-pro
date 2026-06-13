"""
ml_pipeline/train.py

Production ML training pipeline for PhishGuard V2.

Trains and compares three model architectures:
  1. Random Forest     — Robust baseline, interpretable, handles mixed features
  2. XGBoost           — Gradient boosting with regularisation, fast inference
  3. LightGBM          — Histogram-based boosting, fastest training, best on large data

Pipeline stages:
  1. Data loading & validation
  2. Class-imbalance handling with SMOTE (train set only — no leakage)
  3. Bayesian hyperparameter tuning with Optuna (per model)
  4. 5-fold stratified cross-validation
  5. Final model training on full train split
  6. Rich evaluation: Accuracy, F1, Precision, Recall, Log-Loss, ROC-AUC, PR-AUC
  7. Model comparison: side-by-side metrics table + automatic best-model selection
  8. Visualisations: confusion matrices, feature importance, model comparison chart
  9. Secure serialisation (.json for XGBoost, .txt for LightGBM, .joblib for sklearn)

Usage:
  python -m ml_pipeline.train --data ml_pipeline/data/train.csv --output ml_pipeline/models
  python -m ml_pipeline.train --data ml_pipeline/data/train.csv --output ml_pipeline/models --trials 30
  python -m ml_pipeline.train --data ml_pipeline/data/train.csv --test ml_pipeline/data/test.csv
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ml_pipeline.features import URL_FEATURE_COLUMNS, DOM_FEATURE_COLUMNS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phishguard.train")

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
N_CV_FOLDS = 5


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate the training dataset.

    Expected CSV format:
      - Feature columns matching URL_FEATURE_COLUMNS
      - A 'label' column: 1 = phishing, 0 = legitimate
    """
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (1=phishing, 0=legitimate).")

    y = df["label"].astype(int)

    # Determine available feature columns
    all_feature_cols = URL_FEATURE_COLUMNS + DOM_FEATURE_COLUMNS
    available_cols = [c for c in all_feature_cols if c in df.columns]
    missing = [c for c in URL_FEATURE_COLUMNS if c not in df.columns]

    if missing:
        logger.warning(f"Missing URL features (will be zero-filled): {missing}")
        for col in missing:
            df[col] = 0

    X = df[[c for c in URL_FEATURE_COLUMNS if c in df.columns or c in missing]]

    # Fill NaN with 0 (some features may be null for certain URLs)
    X = X.fillna(0)

    logger.info(f"Dataset shape: {X.shape} | Features: {len(X.columns)}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    imbalance_ratio = y.value_counts().get(0, 1) / max(y.value_counts().get(1, 1), 1)
    if imbalance_ratio > 3 or imbalance_ratio < 0.33:
        logger.warning(f"Class imbalance detected (ratio {imbalance_ratio:.2f}). SMOTE will be applied.")

    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
#  SMOTE Resampling
# ═══════════════════════════════════════════════════════════════════════════════

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance. Only called on TRAINING data."""
    logger.info("Applying SMOTE to training set...")
    k = min(5, max(1, min(y_train.value_counts()) - 1))
    if k < 1:
        logger.warning("Too few minority samples for SMOTE — skipping")
        return X_train, y_train
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 1: Random Forest
# ═══════════════════════════════════════════════════════════════════════════════

def tune_random_forest(X_train, y_train, n_trials: int) -> dict:
    """Bayesian hyperparameter tuning for Random Forest via Optuna."""
    logger.info(f"  Tuning Random Forest ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_categorical("max_depth", [None, 10, 20, 30, 50]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 0.9]),
            "class_weight":     "balanced",
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1})
    logger.info(f"  RF best CV F1: {study.best_value:.4f}")
    return best


def train_random_forest(X_train, y_train, params: dict) -> RandomForestClassifier:
    """Train final Random Forest with best hyperparameters."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 2: XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, n_trials: int) -> dict:
    """Bayesian hyperparameter tuning for XGBoost via Optuna."""
    logger.info(f"  Tuning XGBoost ({n_trials} trials)...")

    scale_pos = float(y_train.value_counts().get(0, 1) / max(y_train.value_counts().get(1, 1), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": scale_pos,
            "eval_metric":      "logloss",
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,
            "verbosity":        0,
            "tree_method":      "hist",
            "device":           "cuda",
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "scale_pos_weight": scale_pos,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
        "device": "cuda",
    })
    logger.info(f"  XGB best CV F1: {study.best_value:.4f}")
    return best


def train_xgboost(X_train, y_train, params: dict) -> xgb.XGBClassifier:
    """Train final XGBoost with best hyperparameters."""
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 3: LightGBM
# ═══════════════════════════════════════════════════════════════════════════════

def tune_lightgbm(X_train, y_train, n_trials: int) -> dict:
    """Bayesian hyperparameter tuning for LightGBM via Optuna."""
    logger.info(f"  Tuning LightGBM ({n_trials} trials)...")

    scale_pos = float(y_train.value_counts().get(0, 1) / max(y_train.value_counts().get(1, 1), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 1.0),
            "scale_pos_weight": scale_pos,
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,
            "verbose":          -1,
            "device_type":      "gpu",
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "scale_pos_weight": scale_pos,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
        "device_type": "gpu",
    })
    logger.info(f"  LGBM best CV F1: {study.best_value:.4f}")
    return best


def train_lightgbm(X_train, y_train, params: dict) -> lgb.LGBMClassifier:
    """Train final LightGBM with best hyperparameters."""
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute comprehensive metrics for a single model."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, average="binary"), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "log_loss":  round(log_loss(y_test, y_prob), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
    }

    return metrics, y_pred, y_prob


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    sep = "─" * 50
    logger.info(f"\n{sep}")
    logger.info(f"  {metrics['model']}")
    logger.info(sep)
    logger.info(f"  Accuracy          : {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score          : {metrics['f1']:.4f}")
    logger.info(f"  Precision         : {metrics['precision']:.4f}")
    logger.info(f"  Recall            : {metrics['recall']:.4f}")
    logger.info(f"  Log-Loss          : {metrics['log_loss']:.4f}")
    logger.info(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    logger.info(sep)


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_test, y_pred, model_name: str, output_dir: Path):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fname = _safe_filename(model_name) + "_confusion_matrix.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    logger.info(f"  Confusion matrix → {fname}")


def plot_feature_importance(model, feature_names: list, model_name: str, output_dir: Path):
    """Plot top-20 feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_booster"):
        scores = model.get_booster().get_fscore()
        importances = np.array([scores.get(f"f{i}", 0) for i in range(len(feature_names))], dtype=float)
        total = importances.sum()
        if total > 0:
            importances = importances / total
    else:
        return

    indices = np.argsort(importances)[-20:]
    top_features = np.array(feature_names)[indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(top_features, top_importances, color=colors)
    ax.set_title(f"Top 20 Features — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fname = _safe_filename(model_name) + "_feature_importance.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    logger.info(f"  Feature importance → {fname}")

    # Return ranked feature list
    ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return ranking


def plot_model_comparison(all_metrics: list[dict], output_dir: Path):
    """Side-by-side bar chart comparing all models on key metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metric_keys = ["f1", "precision", "recall", "roc_auc"]
    metric_labels = ["F1 Score", "Precision", "Recall", "ROC-AUC"]
    model_names = [m["model"] for m in all_metrics]
    colors = ["#6366f1", "#10b981", "#f59e0b"]

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        values = [m[key] for m in all_metrics]
        bars = ax.bar(model_names, values, color=colors[:len(model_names)], width=0.6)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Model Comparison — PhishGuard V2", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Model comparison chart → model_comparison.png")


def plot_roc_curves(all_results: list, y_test, output_dir: Path):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#6366f1", "#10b981", "#f59e0b"]

    for (name, _, y_prob), color in zip(all_results, colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    logger.info(f"  ROC curves → roc_curves.png")

def plot_pr_curves(all_results: list, y_test, output_dir: Path):
    """Overlay Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#6366f1", "#10b981", "#f59e0b"]

    for (name, _, y_prob), color in zip(all_results, colors):
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "pr_curves.png", dpi=150)
    plt.close(fig)
    logger.info(f"  PR curves → pr_curves.png")

def generate_shap_plots(model, X_test, model_name: str, output_dir: Path):
    """Generate SHAP summary plot for interpretability."""
    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        # Sample background if dataset is too large
        X_sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values might be a list (one for each class). We want class 1 (phishing)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        fig = plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        fname = f"{_safe_filename(model_name)}_shap_summary.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  SHAP summary plot → {fname}")
    except Exception as e:
        logger.warning(f"  Failed to generate SHAP plot for {model_name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(model, model_name: str, feature_names: list, output_dir: Path):
    """Save model in the most appropriate format."""
    safe_name = _safe_filename(model_name)

    if isinstance(model, xgb.XGBClassifier):
        path = output_dir / f"{safe_name}.json"
        model.save_model(str(path))
    elif isinstance(model, lgb.LGBMClassifier):
        path = output_dir / f"{safe_name}.txt"
        model.booster_.save_model(str(path))
    else:
        path = output_dir / f"{safe_name}.joblib"
        joblib.dump(model, path, compress=3)

    logger.info(f"  Model saved → {path}")

    # Save column order for inference
    cols_path = output_dir / f"{safe_name}_columns.json"
    cols_path.write_text(json.dumps(feature_names, indent=2))

    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Execute the full training pipeline."""
    global N_OPTUNA_TRIALS, N_CV_FOLDS
    N_OPTUNA_TRIALS = args.trials
    N_CV_FOLDS = args.cv_folds

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    X, y = load_data(args.data)
    feature_names = list(X.columns)

    # ── Train/test split ───────────────────────────────────────────────────
    if args.test:
        logger.info(f"Using separate test set: {args.test}")
        X_train, y_train = X, y
        X_test_df = pd.read_csv(args.test)
        y_test = X_test_df["label"].astype(int)
        X_test = X_test_df[[c for c in feature_names if c in X_test_df.columns]].fillna(0)
        for col in feature_names:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[feature_names]
    else:
        logger.info(f"Splitting data: test_size={args.test_size}, stratified=True")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE,
        )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── SMOTE (train only) ─────────────────────────────────────────────────
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── Tune and train all models ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 1: HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    models = {}
    all_params = {}

    # 1. Random Forest (Disabled for massive dataset speed)
    # logger.info("\n Random Forest")
    # rf_params = tune_random_forest(X_train_res, y_train_res, N_OPTUNA_TRIALS)
    # all_params["Random Forest"] = rf_params

    # 2. XGBoost
    logger.info("\n XGBoost")
    xgb_params = tune_xgboost(X_train_res, y_train_res, N_OPTUNA_TRIALS)
    all_params["XGBoost"] = xgb_params

    # 3. LightGBM (Disabled due to GPU tree learner bug on Windows)
    # logger.info("\n LightGBM")
    # lgbm_params = tune_lightgbm(X_train_res, y_train_res, N_OPTUNA_TRIALS)
    # all_params["LightGBM"] = lgbm_params

    # ── Train final models ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 2: FINAL TRAINING")
    logger.info("=" * 60)

    # logger.info("\n Training final Random Forest...")
    # models["Random Forest"] = train_random_forest(X_train_res, y_train_res, rf_params)

    logger.info(" Training final XGBoost...")
    models["XGBoost"] = train_xgboost(X_train_res, y_train_res, xgb_params)

    # logger.info(" Training final LightGBM...")
    # models["LightGBM"] = train_lightgbm(X_train_res, y_train_res, lgbm_params)

    # ── Evaluate all models ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 3: EVALUATION")
    logger.info("=" * 60)

    all_metrics = []
    all_results = []   # (name, y_pred, y_prob)
    all_rankings = {}

    for name, model in models.items():
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        all_results.append((name, y_pred, y_prob))

        print_metrics(metrics)
        logger.info("\n" + classification_report(
            y_test, y_pred, target_names=["Legitimate", "Phishing"]
        ))

        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred, name, output_dir)

        # Feature importance
        ranking = plot_feature_importance(model, feature_names, name, output_dir)
        if ranking:
            all_rankings[name] = ranking

    # ── Model comparison ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 4: MODEL COMPARISON")
    logger.info("=" * 60)

    plot_model_comparison(all_metrics, output_dir)
    plot_roc_curves(all_results, y_test, output_dir)
    plot_pr_curves(all_results, y_test, output_dir)

    # Print comparison table
    comparison_df = pd.DataFrame(all_metrics).set_index("model")
    logger.info(f"\n{comparison_df.to_string()}\n")

    # ── Select best model ──────────────────────────────────────────────────
    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["f1"])
    best_name = all_metrics[best_idx]["model"]
    best_model = models[best_name]
    best_metrics = all_metrics[best_idx]

    logger.info(f" Best model: {best_name} (F1={best_metrics['f1']:.4f})")
    
    # Generate SHAP plot for the best model
    logger.info("  Generating SHAP interpretability plots...")
    generate_shap_plots(best_model, X_test, best_name, output_dir)

    # ── Save all models ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 5: SAVING MODELS")
    logger.info("=" * 60)

    for name, model in models.items():
        save_model(model, name, feature_names, output_dir)

    # Save best model indicator
    best_info = {
        "best_model": best_name,
        "metrics": best_metrics,
        "all_metrics": all_metrics,
        "hyperparameters": {k: _serialize_params(v) for k, v in all_params.items()},
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "smote_samples": len(X_train_res),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "training_report.json").write_text(json.dumps(best_info, indent=2, default=str))
    logger.info(f"  Training report → training_report.json")

    # Save feature rankings
    if all_rankings:
        rankings_output = {}
        for model_name, ranking in all_rankings.items():
            rankings_output[model_name] = [
                {"feature": name, "importance": round(float(imp), 6), "rank": i + 1}
                for i, (name, imp) in enumerate(ranking)
            ]
        (output_dir / "feature_rankings.json").write_text(json.dumps(rankings_output, indent=2))
        logger.info(f"  Feature rankings → feature_rankings.json")

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"   TRAINING COMPLETE ({elapsed:.1f}s)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Best model:    {best_name}")
    logger.info(f"  F1 Score:      {best_metrics['f1']:.4f}")
    logger.info(f"  Precision:     {best_metrics['precision']:.4f}")
    logger.info(f"  Recall:        {best_metrics['recall']:.4f}")
    logger.info(f"  ROC-AUC:       {best_metrics['roc_auc']:.4f}")
    logger.info(f"  Output:        {output_dir}")
    logger.info(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")

def _serialize_params(params: dict) -> dict:
    """Make params JSON-serializable."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="PhishGuard V2 — Train & Compare ML Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on synthetic data (fast)
  python -m ml_pipeline.train --data ml_pipeline/data/synthetic_train.csv --trials 10

  # Train with separate test set (anti-leakage pipeline output)
  python -m ml_pipeline.train --data ml_pipeline/data/train.csv --test ml_pipeline/data/test.csv

  # Full production training (50 Optuna trials)
  python -m ml_pipeline.train --data ml_pipeline/data/train.csv --test ml_pipeline/data/test.csv --trials 50
        """,
    )
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--test", default=None, help="Path to separate test CSV (recommended)")
    parser.add_argument("--output", default="ml_pipeline/models", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (if no --test)")
    parser.add_argument("--trials", type=int, default=N_OPTUNA_TRIALS, help="Optuna trials per model")
    parser.add_argument("--cv-folds", type=int, default=N_CV_FOLDS, help="Cross-validation folds")
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
