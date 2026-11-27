"""
Mix XGBoost and fine-tuned BERT scores → ml_pipeline/models/blend.json

Fits a 2-input logistic regression on V, rows neither model trained on:
XGBoost out-of-fold scores (xgboost_scores.joblib) and BERT scores
(phishbert_scores.joblib). Verdict thresholds are picked on V too.
blend.json is written only if the mix beats XGBoost alone on V; the predictor
uses it when present. Test-set numbers are reported, never used to decide.

Run from the repo root after train.py and finetune_bert.py:
  python -m ml_pipeline.blend
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ml_pipeline.features import BLEND_FILE
from ml_pipeline.train import CUTOFFS, SCORES_FILE, metrics_at, pick_thresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phishguard.blend")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(args):
    models = Path(args.models)
    # joblib is pickle: these are our own train.py / finetune_bert.py outputs, never downloaded files.
    xs, bs =joblib.load(models / SCORES_FILE), joblib.load(models / "phishbert_scores.joblib")
    val = bs["val"].merge(xs["train"], on=["url", "label"], suffixes=("_bert", "_xgb"))  # V rows with OOF XGB
    test = bs["test"].merge(xs["test"], on=["url", "label"], suffixes=("_bert", "_xgb"))
    logger.info(f"V {len(val):,} rows | test {len(test):,} rows")

    feats = lambda d: np.c_[logit(d["score_xgb"]), logit(d["score_bert"])]
    blender = LogisticRegression().fit(feats(val), val["label"])

    report = {}
    for name, p_val, p_test in [
        ("xgboost", val["score_xgb"], test["score_xgb"]),
        ("bert", val["score_bert"], test["score_bert"]),
        ("blend", blender.predict_proba(feats(val))[:, 1], blender.predict_proba(feats(test))[:, 1]),
    ]:
        thr = pick_thresholds(val["label"], p_val, args.phishing_fpr, args.suspicious_fpr)
        report[name] = {
            "thresholds": thr,
            "val_pr_auc": round(average_precision_score(val["label"], p_val), 4),
            "test_pr_auc": round(average_precision_score(test["label"], p_test), 4),
            "test_roc_auc": round(roc_auc_score(test["label"], p_test), 4),
            "test_log_loss": round(log_loss(test["label"], np.clip(p_test, 1e-7, 1 - 1e-7)), 4),
            "test_at": {k: metrics_at(test["label"], p_test, thr[k]) for k in CUTOFFS},
        }
    logger.info("\n" + pd.DataFrame({n: {k: v for k, v in r.items() if "pr_auc" in k or "roc" in k or "loss" in k}
                                     for n, r in report.items()}).to_string())
    for n, r in report.items():
        for k, m in r["test_at"].items():
            logger.info(f"  {n:<8} @{k:<10} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                        f"prec={m['precision']:.4f} rec={m['recall']:.4f} fpr={m['fpr']:.5f}")

    out = models / BLEND_FILE
    if report["blend"]["val_pr_auc"] > report["xgboost"]["val_pr_auc"]:
        out.write_text(json.dumps({
            "coef": blender.coef_[0].tolist(), "intercept": float(blender.intercept_[0]),
            "thresholds": report["blend"]["thresholds"], "report": report,
        }, indent=2))
        logger.info(f"Blend beats XGBoost on V → {out}")
    else:
        out.unlink(missing_ok=True)
        logger.info("Blend does not beat XGBoost on V — not written; predictor stays XGBoost-only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ml_pipeline/models")
    parser.add_argument("--phishing-fpr", type=float, default=0.001)
    parser.add_argument("--suspicious-fpr", type=float, default=0.01)
    main(parser.parse_args())
