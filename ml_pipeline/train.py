"""
ml_pipeline/train.py

Production ML training pipeline for PhishGuard V2.

Trains and compares three model architectures:
  1. Random Forest     — Robust baseline, interpretable, handles mixed features
  2. XGBoost           — Gradient boosting with regularisation, fast inference
  3. LightGBM          — Histogram-based boosting, fastest training, best on large data

Pipeline stages:
  1. Data loading & validation
  2. Domain-grouped splits: no registered domain is in both train and test,
     or in both sides of any CV fold (needs a `url` column)
  3. Bayesian hyperparameter tuning with Optuna, maximising PR-AUC
  4. Out-of-fold scores → thresholds at a target false-positive rate
  5. Final model training on full train split
  6. Rich evaluation at those thresholds: FPR, Recall, Precision, PR-AUC, ...
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
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
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, cross_val_predict, cross_val_score,
)

from ml_pipeline.features import (
    PLATFORMS_FILE, TEXT_FEATURE, TEXT_MODEL_FILE, URL_FEATURE_COLUMNS, abused_hosts, abused_platforms, site_group,
    url_text, url_tokens,
)
SCORES_FILE = "xgboost_scores.joblib"  # OOF train + test scores, for blend.py

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phishguard.train")

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 10  # 30 trials measured no better on held-out data (PR-AUC 0.951 vs 0.952) at 3x the time
N_CV_FOLDS = 5
DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, pd.Series | None]:
    """
    Load and validate the training dataset.

    Expected CSV: URL_FEATURE_COLUMNS, a 'label' column (1 = phishing,
    0 = legitimate), ideally a 'url' column, used to group splits by domain,
    and optionally a 'source' column (which dataset a row came from, for reporting).
    """
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (1=phishing, 0=legitimate).")

    missing = [c for c in URL_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        # Zero-filling silently trains on a stale feature set. Re-extract instead.
        raise ValueError(f"Missing feature columns {missing} — re-run feature extraction.")

    y = df["label"].astype(int)
    X = df[URL_FEATURE_COLUMNS].fillna(0)

    url_col = next((c for c in ("url", "URL") if c in df.columns), None)
    urls = df[url_col].astype(str) if url_col else None
    if urls is None:
        logger.warning(f"No 'url' column in {csv_path} — no domain grouping and no text model.")

    sources = df["source"].fillna("").astype(str) if "source" in df.columns else None
    logger.info(f"Dataset shape: {X.shape} | Class distribution: {y.value_counts().to_dict()}")
    return X, y, urls, sources


def make_cv(groups):
    """Domain-grouped stratified folds when we know the domains."""
    if groups is not None:
        return StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def split_train_test(X, y, groups, test_size: float = 0.2):
    """(train_idx, test_idx): no registered domain on both sides. Deterministic —
    finetune_bert.py and blend.py call this to get the exact same split."""
    n_splits = max(2, round(1 / test_size))
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    if groups is not None:
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        assert not overlap, f"domain leakage: {list(overlap)[:5]}"
    return train_idx, test_idx


def make_text_model():
    """
    Character 3–5-grams + word uni/bigrams (host words tagged) over the normalised
    URL string. Catches patterns the hand-made features can't (paypa1,
    -secure-login-, /wp-admin/.../office365). Hashing keeps memory fixed and the
    saved model small; no vocabulary to fit.
    Chosen on a grouped validation fold: words added +0.012 PR-AUC over chars alone;
    char 2–6 was worse than 3–5.
    """
    hashed = dict(alternate_sign=False, norm=None, dtype=np.float32)
    return make_pipeline(
        make_union(
            HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2**20, **hashed),
            HashingVectorizer(analyzer=url_tokens, n_features=2**20, **hashed),
        ),
        TfidfTransformer(sublinear_tf=True),
        LogisticRegression(C=4.0, solver="liblinear", max_iter=1000),
    )


def noisy_rows(X, y, texts, groups, hi: float = 0.99, lo: float = 0.01) -> np.ndarray:
    """
    Training rows whose label the data itself contradicts: out-of-fold
    (text model → XGBoost) score ≥ hi on a "benign" row, or ≤ lo on a "phishing"
    row. Only ever applied to the TRAIN split; the test set keeps every row.
    """
    cv = make_cv(groups)
    oof_text = cross_val_predict(make_text_model(), texts, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, tree_method="hist",
                            device=DEVICE, random_state=RANDOM_STATE, verbosity=0)
    oof = cross_val_predict(clf, X.assign(**{TEXT_FEATURE: oof_text}), y, groups=groups, cv=cv,
                            method="predict_proba", n_jobs=1 if DEVICE == "cuda" else -1)[:, 1]
    y = np.asarray(y)
    return ((y == 0) & (oof >= hi)) | ((y == 1) & (oof <= lo))


def cv_score(model, X, y, groups) -> float:
    # GPU models must not run folds in parallel processes.
    n_jobs = 1 if DEVICE == "cuda" else -1
    return cross_val_score(model, X, y, groups=groups, cv=make_cv(groups),
                           scoring="average_precision", n_jobs=n_jobs).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 1: Random Forest
# ═══════════════════════════════════════════════════════════════════════════════

def tune_random_forest(X_train, y_train, n_trials: int, groups=None) -> dict:
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
        return cv_score(model, X_train, y_train, groups)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1})
    logger.info(f"  RF best CV PR-AUC: {study.best_value:.4f}")
    return best


def train_random_forest(X_train, y_train, params: dict) -> RandomForestClassifier:
    """Train final Random Forest with best hyperparameters."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 2: XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, n_trials: int, groups=None) -> dict:
    """Bayesian hyperparameter tuning for XGBoost via Optuna."""
    logger.info(f"  Tuning XGBoost ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            "eval_metric":      "logloss",
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,
            "verbosity":        0,
            "tree_method":      "hist",
            "device":           DEVICE,
        }
        model = xgb.XGBClassifier(**params)
        return cv_score(model, X_train, y_train, groups)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
        "device": DEVICE,
    })
    logger.info(f"  XGB best CV PR-AUC: {study.best_value:.4f}")
    return best


def train_xgboost(X_train, y_train, params: dict) -> xgb.XGBClassifier:
    """Train final XGBoost with best hyperparameters."""
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Model 3: LightGBM
# ═══════════════════════════════════════════════════════════════════════════════

def tune_lightgbm(X_train, y_train, n_trials: int, groups=None) -> dict:
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
        return cv_score(model, X_train, y_train, groups)

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
    logger.info(f"  LGBM best CV PR-AUC: {study.best_value:.4f}")
    return best


def train_lightgbm(X_train, y_train, params: dict) -> lgb.LGBMClassifier:
    """Train final LightGBM with best hyperparameters."""
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def pick_thresholds(y, prob, phishing_fpr: float, suspicious_fpr: float) -> dict:
    """
    Thresholds from OUT-OF-FOLD scores: the score above which only `fpr` of
    legitimate URLs fall. A browser extension sees ~1 phish per 1000 pages,
    so the false-positive rate — not F1 at 0.5 — is what users feel.
    """
    y, prob = np.asarray(y), np.asarray(prob)
    benign = prob[y == 0]

    def at(fpr):
        return float(np.nextafter(np.quantile(benign, 1 - fpr, method="higher"), 1.0))

    phishing, suspicious = at(phishing_fpr), at(suspicious_fpr)
    # Max-F1 point: reported only (balanced accuracy/F1); the extension uses the two above.
    prec, rec, cuts = precision_recall_curve(y, prob)
    balanced = float(cuts[np.argmax((2 * prec * rec / np.maximum(prec + rec, 1e-12))[:-1])])
    return {
        "phishing": phishing,
        "suspicious": min(suspicious, phishing),
        "balanced": balanced,
        "phishing_fpr_target": phishing_fpr,
        "suspicious_fpr_target": suspicious_fpr,
        "oof_recall_at_phishing": round(float((prob[y == 1] >= phishing).mean()), 4),
        "oof_recall_at_suspicious": round(float((prob[y == 1] >= suspicious).mean()), 4),
    }


CUTOFFS = ("phishing", "suspicious", "balanced")


def metrics_at(y, prob, t: float) -> dict:
    """Threshold metrics at cut-off t."""
    y, pred = np.asarray(y), (np.asarray(prob) >= t).astype(int)
    return {
        "threshold": round(float(t), 4),
        "fpr":       round(float(pred[y == 0].mean()), 5),
        "accuracy":  round(accuracy_score(y, pred), 4),
        "f1":        round(f1_score(y, pred, zero_division=0), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall":    round(recall_score(y, pred, zero_division=0), 4),
    }


def evaluate_model(model, X_test, y_test, model_name: str, thr: dict) -> dict:
    """Threshold-free metrics, plus threshold metrics at every cut-off in `thr`.
    Top-level threshold metrics are at the deployed 'phishing' cut-off."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= thr["phishing"]).astype(int)

    metrics = {
        "model":     model_name,
        **metrics_at(y_test, y_prob, thr["phishing"]),
        "log_loss":  round(log_loss(y_test, y_prob), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
        "at":        {k: metrics_at(y_test, y_prob, thr[k]) for k in CUTOFFS},
    }

    return metrics, y_pred, y_prob


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    sep = "─" * 50
    logger.info(f"\n{sep}")
    logger.info(f"  {metrics['model']}")
    logger.info(sep)
    logger.info(f"  Threshold         : {metrics['threshold']:.4f}")
    logger.info(f"  False-pos. rate   : {metrics['fpr']:.5f}")
    logger.info(f"  Accuracy          : {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score          : {metrics['f1']:.4f}")
    logger.info(f"  Precision         : {metrics['precision']:.4f}")
    logger.info(f"  Recall            : {metrics['recall']:.4f}")
    logger.info(f"  Log-Loss          : {metrics['log_loss']:.4f}")
    logger.info(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    for k, m in metrics.get("at", {}).items():
        logger.info(f"  @{k:<10} thr={m['threshold']:.4f} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                    f"prec={m['precision']:.4f} rec={m['recall']:.4f} fpr={m['fpr']:.5f}")
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
    global N_OPTUNA_TRIALS, N_CV_FOLDS, DEVICE
    N_OPTUNA_TRIALS = args.trials
    N_CV_FOLDS = args.cv_folds
    DEVICE = args.device

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    X, y, urls, sources = load_data(args.data)
    platforms = abused_platforms(urls, y) if urls is not None else set()
    groups = urls.map(lambda u: site_group(u, platforms)) if urls is not None else None
    hosts = abused_hosts(urls, y) if urls is not None else set()
    (output_dir / PLATFORMS_FILE).write_text(json.dumps({"subdomains": sorted(platforms), "hosts": sorted(hosts)}))
    logger.info(f"{len(platforms)} abused platforms (grouped per host; allowlist skips their subdomains)")

    # ── Train/test split (by registered domain) ────────────────────────────
    if args.test:
        logger.info(f"Using separate test set: {args.test}")
        X_train, y_train, g_train, urls_train = X, y, groups, urls
        X_test, y_test, urls_test, src_test = load_data(args.test)
    else:
        logger.info(f"Splitting data: test_size={args.test_size}, grouped by domain={groups is not None}")
        train_idx, test_idx = split_train_test(X, y, groups, args.test_size)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups.iloc[train_idx] if groups is not None else None
        urls_train = urls.iloc[train_idx] if urls is not None else None
        urls_test = urls.iloc[test_idx] if urls is not None else None
        src_test = sources.iloc[test_idx] if sources is not None else None

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Optional: drop training rows whose labels the data contradicts ─────
    pruned_urls = []
    if args.prune_noise and urls_train is not None:
        logger.info("Finding contradicted training labels (out-of-fold)...")
        drop = noisy_rows(X_train, y_train, [url_text(u) for u in urls_train], g_train)
        pruned_urls = urls_train[drop].tolist()
        logger.info(f"  Dropping {drop.sum():,} training rows: "
                    f"{int((drop & (y_train.values == 0)).sum()):,} 'benign', "
                    f"{int((drop & (y_train.values == 1)).sum()):,} 'phishing'")
        keep = ~drop
        X_train, y_train, urls_train = X_train[keep], y_train[keep], urls_train[keep]
        g_train = g_train[keep] if g_train is not None else None

    # ── Stage-1 text model, stacked into XGBoost as one more feature ───────
    # Train rows get OUT-OF-FOLD text scores (same domain-grouped folds), so
    # XGBoost never sees a score the text model produced on its own training rows.
    text_model, text_report = None, {"used": False}
    if args.no_text:
        logger.info("Text model disabled (--no-text).")
    elif urls_train is None or urls_test is None:
        logger.warning("Train or test CSV has no 'url' column — skipping the text model.")
    else:
        logger.info("Fitting n-gram text model (out-of-fold on train)...")
        texts_train = [url_text(u) for u in urls_train]
        texts_test = [url_text(u) for u in urls_test]
        oof_text = cross_val_predict(
            make_text_model(), texts_train, y_train, groups=g_train,
            cv=make_cv(g_train), method="predict_proba",
        )[:, 1]
        text_model = make_text_model().fit(texts_train, y_train)
        X_train = X_train.assign(**{TEXT_FEATURE: oof_text})
        X_test = X_test.assign(**{TEXT_FEATURE: text_model.predict_proba(texts_test)[:, 1]})
        text_report = {
            "used": True,
            "oof_pr_auc_text_only": round(average_precision_score(y_train, oof_text), 4),
            "test_pr_auc_text_only": round(average_precision_score(y_test, X_test[TEXT_FEATURE]), 4),
        }
        logger.info(f"  Text model alone: {text_report}")

    feature_names = list(X_train.columns)

    # ── Tune and train all models ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 1: HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    models = {}
    all_params = {}

    # 1. Random Forest (Disabled for massive dataset speed)
    # logger.info("\n Random Forest")
    # rf_params = tune_random_forest(X_train, y_train, N_OPTUNA_TRIALS, g_train)
    # all_params["Random Forest"] = rf_params

    # 2. XGBoost
    logger.info("\n XGBoost")
    xgb_params = tune_xgboost(X_train, y_train, N_OPTUNA_TRIALS, g_train)
    all_params["XGBoost"] = xgb_params

    # 3. LightGBM (Disabled due to GPU tree learner bug on Windows)
    # logger.info("\n LightGBM")
    # lgbm_params = tune_lightgbm(X_train, y_train, N_OPTUNA_TRIALS, g_train)
    # all_params["LightGBM"] = lgbm_params

    # ── Train final models ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 2: FINAL TRAINING")
    logger.info("=" * 60)

    # logger.info("\n Training final Random Forest...")
    # models["Random Forest"] = train_random_forest(X_train, y_train, rf_params)

    logger.info(" Training final XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train, xgb_params)

    # logger.info(" Training final LightGBM...")
    # models["LightGBM"] = train_lightgbm(X_train, y_train, lgbm_params)

    # ── Evaluate all models ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 3: EVALUATION")
    logger.info("=" * 60)

    all_metrics = []
    all_results = []   # (name, y_pred, y_prob)
    all_rankings = {}
    all_thresholds = {}
    all_oof = {}

    for name, model in models.items():
        # Thresholds come from out-of-fold train scores, never from the test set.
        oof = cross_val_predict(
            clone(model), X_train, y_train, groups=g_train, cv=make_cv(g_train),
            method="predict_proba", n_jobs=1 if DEVICE == "cuda" else -1,
        )[:, 1]
        thr = pick_thresholds(y_train, oof, args.phishing_fpr, args.suspicious_fpr)
        all_thresholds[name], all_oof[name] = thr, oof
        logger.info(f"  {name} thresholds: {thr}")

        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name, thr)
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
    comparison_df = pd.DataFrame(all_metrics).drop(columns="at").set_index("model")
    logger.info(f"\n{comparison_df.to_string()}\n")

    # ── Select best model ──────────────────────────────────────────────────
    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["pr_auc"])
    best_name = all_metrics[best_idx]["model"]
    best_model = models[best_name]
    best_metrics = all_metrics[best_idx]

    logger.info(f" Best model: {best_name} (PR-AUC={best_metrics['pr_auc']:.4f})")

    # Per source dataset: label quality differs a lot between them.
    by_source = {}
    if src_test is not None:
        best_prob = dict((n, p) for n, _, p in all_results)[best_name]
        yt, st = np.asarray(y_test), np.asarray(src_test)
        for s in sorted(set(st)):
            m = st == s
            by_source[s] = {"benign": int((m & (yt == 0)).sum()), "phishing": int((m & (yt == 1)).sum()), **{
                f"{k}_{'recall' if c else 'fpr'}": round(float((best_prob[m & (yt == c)] >= all_thresholds[best_name][k]).mean()), 4)
                for k in ("phishing", "suspicious") for c in (1, 0) if (m & (yt == c)).any()}}
        logger.info("  By source:\n" + pd.DataFrame(by_source).T.to_string())

    # Generate SHAP plot for the best model
    logger.info("  Generating SHAP interpretability plots...")
    generate_shap_plots(best_model, X_test, best_name, output_dir)

    # ── Save all models ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 5: SAVING MODELS")
    logger.info("=" * 60)

    if text_model is not None:
        joblib.dump(text_model, output_dir / TEXT_MODEL_FILE, compress=3)
        logger.info(f"  Text model → {output_dir / TEXT_MODEL_FILE}")
    elif (output_dir / TEXT_MODEL_FILE).exists():
        (output_dir / TEXT_MODEL_FILE).unlink()  # stale: the new XGBoost doesn't expect it

    for name, model in models.items():
        save_model(model, name, feature_names, output_dir)
        thr_path = output_dir / f"{_safe_filename(name)}_thresholds.json"
        thr_path.write_text(json.dumps(all_thresholds[name], indent=2))
        logger.info(f"  Thresholds → {thr_path}")

    # Out-of-fold train scores + test scores of the best model, for blend.py.
    if urls_train is not None and urls_test is not None:
        best_prob = dict((n, p) for n, _, p in all_results)[best_name]
        joblib.dump({
            "train": pd.DataFrame({"url": urls_train.values, "label": y_train.values, "score": all_oof[best_name]}),
            "test": pd.DataFrame({"url": urls_test.values, "label": y_test.values, "score": best_prob}),
            "pruned_urls": pruned_urls,
        }, output_dir / SCORES_FILE, compress=3)

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
        "thresholds": all_thresholds,
        "domain_grouped": g_train is not None,
        "text_model": text_report,
        "pruned_training_rows": len(pruned_urls),
        "by_source": by_source,
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
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="XGBoost device")
    parser.add_argument("--phishing-fpr", type=float, default=0.001,
                        help="Target false-positive rate for the 'phishing' verdict (default 0.1%%)")
    parser.add_argument("--suspicious-fpr", type=float, default=0.01,
                        help="Target false-positive rate for the 'suspicious' verdict (default 1%%)")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip the n-gram text model (e.g. to compare, or if RAM is tight)")
    parser.add_argument("--prune-noise", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop training rows whose label out-of-fold scores contradict (test untouched). "
                             "On by default: +0.003 PR-AUC, +4 pts recall at 1%% FP on held-out data")
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
