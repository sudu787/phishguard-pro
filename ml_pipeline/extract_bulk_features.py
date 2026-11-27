"""Extract URL features for megaset.csv → megaset_features.csv (keeps `url` for domain-grouped CV)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import time
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from ml_pipeline.features import URLFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

extractor = URLFeatureExtractor()


def process_chunk(chunk):
    results = []
    for url, label, source in chunk:
        feats = extractor.extract(str(url))
        feats['label'] = label
        feats['url'] = url
        feats['source'] = source
        results.append(feats)
    return results


def main():
    df = pd.read_csv('ml_pipeline/data/megaset.csv')
    rows = list(zip(df['url'], df['label'], df.get('source', pd.Series('', index=df.index))))
    logger.info(f"Total URLs to process: {len(rows)}")

    # No reuse of older feature CSVs: they were built with a different feature set.
    n = cpu_count()
    size = len(rows) // n + 1
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]

    t0 = time.time()
    out = []
    with Pool(n) as p:
        for res in tqdm(p.imap_unordered(process_chunk, chunks), total=len(chunks)):
            out.extend(res)
    logger.info(f"Extraction took {time.time() - t0:.1f}s")

    pd.DataFrame(out).to_csv('ml_pipeline/data/megaset_features.csv', index=False)
    logger.info(f"Saved {len(out)} rows to megaset_features.csv")


if __name__ == "__main__":
    main()
