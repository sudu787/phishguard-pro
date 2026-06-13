import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from ml_pipeline.features import URLFeatureExtractor
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

extractor = URLFeatureExtractor()

def process_chunk(chunk):
    results = []
    for url, label in chunk:
        try:
            feats = extractor.extract(url)
            feats['label'] = label
            feats['URL'] = url
            results.append(feats)
        except Exception:
            pass
    return results

def main():
    logger.info("Loading megaset.csv...")
    df = pd.read_csv('ml_pipeline/data/megaset.csv')
    
    urls_and_labels = list(zip(df['url'], df['label']))
    logger.info(f"Total URLs to process: {len(urls_and_labels)}")
    
    # Check if kaggle_features.csv has some already
    try:
        kaggle_df = pd.read_csv('ml_pipeline/data/kaggle_features.csv')
        existing_urls = set(kaggle_df['URL']) if 'URL' in kaggle_df.columns else set()
        logger.info(f"Found {len(existing_urls)} existing URLs.")
    except Exception:
        existing_urls = set()

    # Filter out existing
    to_process = [x for x in urls_and_labels if x[0] not in existing_urls]
    logger.info(f"URLs needing extraction: {len(to_process)}")
    
    # We will use simple multithreading or multiprocessing
    num_cores = cpu_count()
    chunk_size = len(to_process) // num_cores + 1
    chunks = [to_process[i:i + chunk_size] for i in range(0, len(to_process), chunk_size)]
    
    logger.info(f"Starting multiprocessing with {num_cores} cores...")
    start_time = time.time()
    
    all_features = []
    with Pool(num_cores) as p:
        for res in tqdm(p.imap_unordered(process_chunk, chunks), total=len(chunks)):
            all_features.extend(res)
            
    logger.info(f"Extraction took {time.time() - start_time:.1f}s")
    
    new_df = pd.DataFrame(all_features)
    
    if len(existing_urls) > 0:
        logger.info("Combining with existing features...")
        final_df = pd.concat([kaggle_df, new_df], ignore_index=True)
    else:
        final_df = new_df
        
    # Deduplicate again just in case
    if 'URL' in final_df.columns:
        final_df = final_df.drop_duplicates(subset=['URL'])
        
    logger.info(f"Saving {len(final_df)} rows to megaset_features.csv")
    final_df.to_csv('ml_pipeline/data/megaset_features.csv', index=False)
    logger.info("Done!")

if __name__ == "__main__":
    main()
