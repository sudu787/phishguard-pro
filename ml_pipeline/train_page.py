"""
Tier 2 (URL + page) model → ml_pipeline/models/{page_text_model.joblib, page_xgb.json, page_blend.json}

Page model = XGBoost on the extension's DOM features + a stacked TF-IDF score of the page text
(title + visible text). Final Tier 2 score = logistic mix of [logit URL score, logit page score].

Reuses the URL model's split so nothing leaks (pages from fetch_phreshphish_html.py, PhreshPhish TRAIN only):
  page model  ← pages in the URL model's train part, minus its validation fold V
  mix + cut-offs ← pages in V (URL and page scores both out-of-sample)
  report      ← pages in the URL model's held-out test
With --bert it also fine-tunes DistilBERT on the page text (GPU, ~3 h) as a third mix input; its V/test
scores are cached in pagebert_scores.joblib and reused on later runs.
Run after train.py / finetune_bert.py / blend.py:  python -m ml_pipeline.train_page [--bert]
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from ml_pipeline.features import (
    BLEND_FILE, DOM_FEATURE_COLUMNS, PAGE_BERT_DIR, PAGE_BERT_MAX_LEN, PAGE_BLEND_FILE, PAGE_TEXT_MODEL_FILE, PAGE_TEXT_SCORE, PAGE_XGB_FILE,
    PLATFORMS_FILE, site_group,
)
from ml_pipeline.train import CUTOFFS, RANDOM_STATE, SCORES_FILE, metrics_at, pick_thresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phishguard.train_page")

MODELS = Path("ml_pipeline/models")
XGB_PARAMS = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "eval_metric": "aucpr", "tree_method": "hist", "random_state": RANDOM_STATE}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_page_text_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True),
                         LogisticRegression(C=4, max_iter=2000))


def url_scores() -> pd.DataFrame:
    """URL-model blend scores for V and test rows (both out-of-sample), tagged with their part."""
    # joblib = pickle: our own train.py / finetune_bert.py outputs, never downloaded files.
    xs, bs = joblib.load(MODELS / SCORES_FILE), joblib.load(MODELS / "phishbert_scores.joblib")
    bl = json.loads((MODELS / BLEND_FILE).read_text())
    parts = []
    for part, b, x in [("V", bs["val"], xs["train"]), ("test", bs["test"], xs["test"])]:
        d = b.merge(x, on=["url", "label"], suffixes=("_bert", "_xgb"))
        z = bl["intercept"] + bl["coef"][0] * logit(d.score_xgb) + bl["coef"][1] * logit(d.score_bert)
        parts.append(pd.DataFrame({"url": d.url, "url_score": 1 / (1 + np.exp(-z)), "part": part}))
    return pd.concat(parts).drop_duplicates("url")


def page_scores(text_model, booster, d: pd.DataFrame) -> np.ndarray:
    X = d[DOM_FEATURE_COLUMNS].assign(**{PAGE_TEXT_SCORE: text_model.predict_proba(d.page_text)[:, 1]})
    return booster.predict(xgb.DMatrix(X.values, feature_names=list(X.columns)))


def evaluate(y, p, thr) -> dict:
    ben = np.sort(p[y == 0]); fixed = lambda f: ben[int((1 - f) * len(ben))]
    return {"pr_auc": round(average_precision_score(y, p), 4), "roc_auc": round(roc_auc_score(y, p), 4),
            "rec@0.1%FP": round(float((p[y == 1] > fixed(0.001)).mean()), 4),
            "rec@1%FP": round(float((p[y == 1] > fixed(0.01)).mean()), 4),
            **{f"at_{k}": metrics_at(y, p, thr[k]) for k in CUTOFFS}}


def main(args):
    d = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(Path(args.pages).glob("train-*.csv"))], ignore_index=True)
    d["y"] = (d.label == "phish").astype(int)
    plat = set(json.loads((MODELS / PLATFORMS_FILE).read_text())["subdomains"])
    d["grp"] = d.url.map(lambda u: site_group(u, plat))
    d = d.merge(url_scores(), on="url", how="left")
    d["part"] = d.part.fillna("train")
    held = set(d.grp[d.part != "train"])
    d = d[(d.part != "train") | ~d.grp.isin(held)]  # page-train never shares a site with V/test
    tr, va, te = (d[d.part == p].reset_index(drop=True) for p in ("train", "V", "test"))
    logger.info(f"pages: train {len(tr):,} | V {len(va):,} | test {len(te):,} (phish {tr.y.mean():.0%}/{va.y.mean():.0%}/{te.y.mean():.0%})")

    # Stacked page-text score: out-of-fold on page-train so XGBoost sees honest scores.
    oof = np.zeros(len(tr))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(tr, tr.y, tr.grp):
        oof[b] = make_page_text_model().fit(tr.page_text.iloc[a], tr.y.iloc[a]).predict_proba(tr.page_text.iloc[b])[:, 1]
    text_model = make_page_text_model().fit(tr.page_text, tr.y)
    Xtr = tr[DOM_FEATURE_COLUMNS].assign(**{PAGE_TEXT_SCORE: oof})
    booster = xgb.XGBClassifier(**XGB_PARAMS).fit(Xtr, tr.y).get_booster()

    pv, pt = page_scores(text_model, booster, va), page_scores(text_model, booster, te)
    bert_dir = MODELS / args.bert_dir
    cache = MODELS / f"{args.bert_dir}_scores.joblib"
    if args.bert:
        from ml_pipeline.finetune_bert import fine_tune, score
        model, tok, device, _ = fine_tune(list(tr.page_text), tr.y.values, list(va.page_text), va.y.values,
                                          bert_dir, args.epochs, args.batch, args.lr, args.max_len,
                                          base=args.bert_base, freeze_layers=args.freeze_layers)
        bert = lambda df: score(model, tok, list(df.page_text), device, batch=64, max_len=args.max_len)
        joblib.dump(pd.concat([pd.DataFrame({"sha256": df.sha256, "bert": bert(df)}) for df in (va, te)]), cache)
    use_bert = cache.exists() and bert_dir.exists()
    bv = bt = None
    if use_bert:
        bs = joblib.load(cache).set_index("sha256")["bert"]  # our own output above
        bv, bt = bs.reindex(va.sha256).values, bs.reindex(te.sha256).values
    # Mix inputs: [logit URL, logit page] (+ logit page BERT) — the order the predictor uses.
    feats = lambda df, p, b: np.column_stack([logit(df.url_score), logit(p)] + ([logit(b)] if use_bert else []))
    mix = LogisticRegression().fit(feats(va, pv, bv), va.y)
    sv, st = mix.predict_proba(feats(va, pv, bv))[:, 1], mix.predict_proba(feats(te, pt, bt))[:, 1]

    report = {}
    rows = [("url", va.url_score.values, te.url_score.values), ("page", pv, pt)]
    rows += [("page_bert", bv, bt)] if use_bert else []
    for name, v, t in rows + [("url+page", sv, st)]:
        thr = pick_thresholds(va.y, v, 0.001, 0.01)
        report[name] = {"val_pr_auc": round(average_precision_score(va.y, v), 4), "thresholds": thr, "test": evaluate(te.y.values, t, thr)}
        r = report[name]["test"]
        logger.info(f"{name:<9} V PR-AUC {report[name]['val_pr_auc']:.4f} | test PR-AUC {r['pr_auc']:.4f} ROC {r['roc_auc']:.4f} "
                    f"rec@0.1%FP {r['rec@0.1%FP']:.4f} rec@1%FP {r['rec@1%FP']:.4f}")
        for k in CUTOFFS:
            m = r[f"at_{k}"]
            logger.info(f"    @{k:<10} acc={m['accuracy']:.4f} f1={m['f1']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} fpr={m['fpr']:.5f}")

    if report["url+page"]["val_pr_auc"] <= report["url"]["val_pr_auc"]:
        logger.info("URL+page does not beat URL alone on V — Tier 2 not written.")
        return
    joblib.dump(text_model, MODELS / PAGE_TEXT_MODEL_FILE)
    booster.save_model(MODELS / PAGE_XGB_FILE)
    (MODELS / PAGE_BLEND_FILE).write_text(json.dumps({
        "coef": mix.coef_[0].tolist(), "bert_max_len": args.max_len if use_bert else None, "intercept": float(mix.intercept_[0]),
        "thresholds": report["url+page"]["thresholds"], "report": report}, indent=2))
    logger.info(f"Tier 2 beats URL alone on V → {MODELS / PAGE_BLEND_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", action="store_true", help="fine-tune DistilBERT on page text (GPU)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bert-base", default="distilbert-base-uncased", help="HF model to fine-tune (safetensors only)")
    ap.add_argument("--bert-dir", default=PAGE_BERT_DIR, help="output dir under models/ (the predictor loads pagebert/)")
    ap.add_argument("--pages", default="ml_pipeline/data/phreshphish/pages", help="fetch_phreshphish_html.py output")
    ap.add_argument("--max-len", type=int, default=PAGE_BERT_MAX_LEN, help="page BERT tokens (saved in page_blend.json)")
    ap.add_argument("--freeze-layers", type=int, default=0, help="freeze embeddings + first N layers (BERT-large on 6 GB)")
    main(ap.parse_args())
