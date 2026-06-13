import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading malicious_phish.csv...")
    df1 = pd.read_csv('ml_pipeline/data/malicious_phish.csv')
    df1['label'] = df1['type'].apply(lambda x: 0 if x in ['benign', 'legitimate'] else 1)
    df1 = df1[['url', 'label']]
    logger.info(f"Loaded {len(df1)} rows from malicious_phish.csv")

    logger.info("Loading url_dataset.csv...")
    df2 = pd.read_csv('ml_pipeline/data/url_dataset.csv')
    df2['label'] = df2['type'].apply(lambda x: 0 if x in ['benign', 'legitimate'] else 1)
    df2 = df2[['url', 'label']]
    logger.info(f"Loaded {len(df2)} rows from url_dataset.csv")

    logger.info("Loading PhiUSIIL_Phishing_URL_Dataset.csv...")
    df3 = pd.read_csv('ml_pipeline/data/PhiUSIIL_Phishing_URL_Dataset.csv', usecols=['URL', 'label'])
    df3 = df3.rename(columns={'URL': 'url'})
    # In PhiUSIIL, 1 is legitimate and 0 is phishing!
    # Wait, earlier we found that 1 is benign! So we invert it so 1 is phishing, 0 is benign.
    df3['label'] = 1 - df3['label']
    logger.info(f"Loaded {len(df3)} rows from PhiUSIIL")

    logger.info("Combining datasets...")
    dataframes = [df1, df2, df3]
    
    # Phase 5: Integrate user feedback reports
    import os
    reports_path = 'ml_pipeline/data/user_reports.csv'
    if os.path.exists(reports_path):
        logger.info("Loading user_reports.csv (Phase 5 Feedback Loop)...")
        df_reports = pd.read_csv(reports_path)
        logger.info(f"Loaded {len(df_reports)} manual corrections from users")
        dataframes.append(df_reports)
    else:
        logger.info("No user_reports.csv found. Skipping Phase 5 Feedback.")

    megaset = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total rows before deduplication: {len(megaset)}")

    # Drop duplicates
    megaset = megaset.drop_duplicates(subset=['url'])
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
