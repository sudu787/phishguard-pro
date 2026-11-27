"""
Fine-tune DistilBERT on normalised URLs → ml_pipeline/models/phishbert/ + phishbert_scores.joblib

Uses the exact domain-grouped train/test split of train.py (split_train_test).
Inside the train part, the first make_cv fold is V: BERT trains on train − V and
is scored on V and on test, so blend.py can fit the XGBoost/BERT mix on rows
neither model trained on. Rows train.py pruned as label noise are dropped too,
so run this AFTER the train.py run whose models you'll ship.

Input is url_text(url), the same normalised string the predictor sees.

Run from the repo root (GPU):
  python -m ml_pipeline.finetune_bert --data ml_pipeline/data/megaset_features.csv
"""
import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from ml_pipeline import train as T
from ml_pipeline.features import BERT_DIR, BERT_MAX_LEN, abused_platforms, site_group, url_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phishguard.bert")

BASE = "distilbert-base-uncased"


def loader(tok, texts, labels, batch, shuffle=False, max_len=BERT_MAX_LEN):
    enc = tok(texts, truncation=True, max_length=max_len)
    items = [{"input_ids": i, "attention_mask": m, "labels": int(l)}
             for i, m, l in zip(enc["input_ids"], enc["attention_mask"], labels)]
    return DataLoader(items, batch_size=batch, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tok))


@torch.inference_mode()
def score(model, tok, texts, device, batch=256, max_len=BERT_MAX_LEN) -> np.ndarray:
    model.eval()
    out = []
    for b in loader(tok, texts, [0] * len(texts), batch, max_len=max_len):
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def fine_tune(fit_texts, fit_labels, val_texts, val_labels, out_dir, epochs, batch, lr, max_len=BERT_MAX_LEN,
              base=BASE, freeze_layers=0):
    """Fine-tune `base` (label 1 = phishing), keep the epoch with the best V PR-AUC in out_dir
    → (model, tok, device, best V PR-AUC). freeze_layers > 0 freezes the embeddings and the first N encoder
    layers and turns on gradient checkpointing: how BERT-large fits a 6 GB GPU without an 8-bit optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(device)
    if freeze_layers:
        enc = model.base_model
        for p in [*enc.embeddings.parameters(), *(q for layer in enc.encoder.layer[:freeze_layers] for q in layer.parameters())]:
            p.requires_grad = False
        model.gradient_checkpointing_enable()
    train_dl = loader(tok, fit_texts, fit_labels, batch, shuffle=True, max_len=max_len)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best = -1.0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for step, b in enumerate(train_dl):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 1000 == 0:
                logger.info(f"  epoch {ep + 1} step {step}/{len(train_dl)} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        ap = average_precision_score(val_labels, score(model, tok, val_texts, device, max_len=max_len))
        logger.info(f"Epoch {ep + 1}: V PR-AUC {ap:.4f} ({time.time() - t0:.0f}s)")
        if ap > best:
            best = ap
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    return AutoModelForSequenceClassification.from_pretrained(out_dir).to(device), tok, device, best  # best epoch


def main(args):
    models_dir = Path(args.models)
    X, y, urls, _ = T.load_data(args.data)
    platforms = abused_platforms(urls, y)  # same groups as train.py
    groups = urls.map(lambda u: site_group(u, platforms))
    tr, te = T.split_train_test(X, y, groups)
    _, v = next(T.make_cv(groups.iloc[tr]).split(X.iloc[tr], y.iloc[tr], groups.iloc[tr]))
    val_idx = tr[v]
    scores_path = models_dir / T.SCORES_FILE  # our own train.py output (joblib = pickle: never load downloaded ones)
    pruned =set(joblib.load(scores_path)["pruned_urls"]) if scores_path.exists() else set()
    fit_idx = np.setdiff1d(tr, val_idx)
    fit_idx = fit_idx[~urls.iloc[fit_idx].isin(pruned).values]
    logger.info(f"BERT fit {len(fit_idx):,} | V {len(val_idx):,} | test {len(te):,} (pruned {len(pruned):,})")

    texts = np.array([url_text(u) for u in urls], dtype=object)
    labels = y.values
    out_dir = models_dir / BERT_DIR
    model, tok, device, best = fine_tune(list(texts[fit_idx]), labels[fit_idx], list(texts[val_idx]), labels[val_idx],
                                         out_dir, args.epochs, args.batch, args.lr)
    frame = lambda idx: pd.DataFrame({"url": urls.iloc[idx].values, "label": labels[idx],
                                      "score": score(model, tok, list(texts[idx]), device)})
    test = frame(te)
    joblib.dump({"val": frame(val_idx), "test": test, "val_pr_auc": best}, models_dir / "phishbert_scores.joblib", compress=3)
    logger.info(f"Saved {out_dir} (V PR-AUC {best:.4f}, test PR-AUC "
                f"{average_precision_score(test['label'], test['score']):.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ml_pipeline/data/megaset_features.csv")
    parser.add_argument("--models", default="ml_pipeline/models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    main(parser.parse_args())
