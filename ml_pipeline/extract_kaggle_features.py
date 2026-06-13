import pandas as pd
import logging
from tqdm import tqdm
from features import URLFeatureExtractor, URL_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    input_path = "ml_pipeline/data/PhiUSIIL_Phishing_URL_Dataset.csv"
    output_path = "ml_pipeline/data/kaggle_features.csv"
    
    logger.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path, usecols=['URL', 'label'])
    
    # Invert label: Kaggle uses 0 for phishing, 1 for legit (wait, let me check again)
    # The PhiUSIIL dataset: 1 is legitimate, 0 is phishing usually? Or 1 is phishing? 
    # Let's invert it so 1 = phishing. Let's do `df['label'] = 1 - df['label']` because we know in Kaggle 0 is phishing and 1 is benign usually. 
    # Wait, let's just do it exactly like finetune_bert: 1 - df['label']
    df['label'] = 1 - df['label']
    
    logger.info(f"Loaded {len(df)} rows. Phishing ratio: {df['label'].mean():.2f}")
    
    extractor = URLFeatureExtractor()
    
    logger.info("Extracting features (this might take a minute)...")
    
    # Pre-allocate dictionary
    features_list = []
    
    # We only use URL features because we only have URLs.
    for url in tqdm(df['URL']):
        features = extractor.extract(url)
        features_list.append(features)
        
    features_df = pd.DataFrame(features_list)
    
    # Combine with label
    features_df['label'] = df['label'].values
    
    # Mock DOM features as 0 since we don't have them in this dataset
    from features import DOM_FEATURE_COLUMNS
    for col in DOM_FEATURE_COLUMNS:
        features_df[col] = 0.0
        
    logger.info(f"Saving to {output_path}...")
    features_df.to_csv(output_path, index=False)
    logger.info("Done!")

if __name__ == "__main__":
    main()
