import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS = {'benign': 0, 'legitimate': 0, 'phishing': 1}


def load_typed(path):
    """Keep only phishing/benign rows. malware & defacement are different
    problems — labelling them phishing teaches the model the wrong thing."""
    df = pd.read_csv(path)
    dropped = df.loc[~df['type'].isin(LABELS), 'type'].value_counts().to_dict()
    df = df[df['type'].isin(LABELS)].copy()
    df['label'] = df['type'].map(LABELS)
    logger.info(f"Loaded {len(df)} rows from {path} (dropped {dropped})")
    return df[['url', 'label']].assign(source=Path(path).stem)


def main():
    df1 = load_typed('ml_pipeline/data/malicious_phish.csv')
    # Its scheme-less "phishing" rows are mostly ordinary sites (university pages,
    # news articles) mislabelled at source; the ones with a scheme are real kits.
    # Features never see the scheme, so filtering on it adds no shortcut.
    noisy = (df1['label'] == 1) & ~df1['url'].astype(str).str.contains('://', regex=False)
    df1 = df1[~noisy]
    logger.info(f"Dropped {noisy.sum()} scheme-less 'phishing' rows from malicious_phish (mislabelled)")
    df2 = load_typed('ml_pipeline/data/url_dataset.csv')

    logger.info("Loading PhiUSIIL_Phishing_URL_Dataset.csv...")
    df3 = pd.read_csv('ml_pipeline/data/PhiUSIIL_Phishing_URL_Dataset.csv', usecols=['URL', 'label'])
    df3 = df3.rename(columns={'URL': 'url'})
    # In PhiUSIIL, 1 is legitimate and 0 is phishing!
    # Wait, earlier we found that 1 is benign! So we invert it so 1 is phishing, 0 is benign.
    df3['label'] = 1 - df3['label']
    df3['source'] = 'PhiUSIIL'
    logger.info(f"Loaded {len(df3)} rows from PhiUSIIL")

    logger.info("Combining datasets...")
    dataframes = [df1, df2, df3]

    # Diverse legitimate deep links (fetch_commoncrawl_benign.py), if fetched.
    # Not benign pages ON user-content hosts: tried, it lowered held-out PR-AUC
    # (0.932 → 0.923) — from the URL alone they look like the phishing there.
    cc_path = Path('ml_pipeline/data/commoncrawl_benign.csv')
    if cc_path.exists():
        dataframes.append(load_typed(cc_path))

    # PhreshPhish (CC-BY-4.0, 2024-25, ml_pipeline/fetch_phreshphish.py): TRAIN split only — its
    # test split is the external benchmark and must never be trained on.
    pp_path = Path('ml_pipeline/data/phreshphish/train.csv')
    if pp_path.exists():
        df_pp = pd.read_csv(pp_path, usecols=['url', 'label'])
        df_pp['label'] = df_pp['label'].map({'benign': 0, 'phish': 1})
        dataframes.append(df_pp.dropna().astype({'label': int}).assign(source='phreshphish'))
        logger.info(f"Loaded {len(df_pp)} rows from PhreshPhish train")

    # Phase 5: Integrate user feedback reports
    import os
    reports_path = 'ml_pipeline/data/user_reports.csv'
    if os.path.exists(reports_path):
        logger.info("Loading user_reports.csv (Phase 5 Feedback Loop)...")
        df_reports = pd.read_csv(reports_path).assign(source='user_reports')
        logger.info(f"Loaded {len(df_reports)} manual corrections from users")
        dataframes.append(df_reports)
    else:
        logger.info("No user_reports.csv found. Skipping Phase 5 Feedback.")

    megaset = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total rows before deduplication: {len(megaset)}")

    # Dedup on the normalised form, so "x.com/a" and "https://www.x.com/a/" collapse
    from ml_pipeline.features import normalise
    megaset['_norm'] = megaset['url'].astype(str).map(lambda u: normalise(u)[0].lower())
    # A URL labelled both ways is noise either way: drop it rather than keep whichever came first.
    conflict = megaset.groupby('_norm')['label'].transform('nunique') > 1
    logger.info(f"Dropping {conflict.sum()} rows whose URL has conflicting labels")
    megaset = megaset[~conflict].drop_duplicates(subset=['_norm']).drop(columns='_norm')
    logger.info(f"Total rows after deduplication: {len(megaset)}")

    # Print label distribution
    dist = megaset['label'].value_counts()
    logger.info(f"Label distribution:\n{dist}")

    # Save to disk
    logger.info("Saving megaset.csv...")
    megaset.to_csv('ml_pipeline/data/megaset.csv', index=False)
    logger.info("Done!")

if __name__ == '__main__':
    main()
