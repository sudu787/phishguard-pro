import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class URLDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main(args):
    model_name = "distilbert-base-uncased"
    output_dir = "ml_pipeline/models/phishbert"

    logger.info(f"Loading dataset from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    if 'URL' in df.columns:
        df = df.rename(columns={'URL': 'url'})
        
    logger.info(f"Loaded {len(df)} total rows.")
    
    # Invert labels ONLY if we are using the original PhiUSIIL dataset.
    # The megaset already has 1=Phishing, 0=Benign.
    if 'PhiUSIIL' in args.data_path:
        logger.info("Inverting PhiUSIIL labels to PhishGuard format (1=Phishing, 0=Benign).")
        df['label'] = 1 - df['label']
    
    if args.sample_size > 0 and len(df) > args.sample_size:
        logger.info(f"Sampling {args.sample_size} rows to save time...")
        df = df.sample(n=args.sample_size, random_state=42)

    logger.info(f"Dataset size: {len(df)} rows. Phishing ratio: {df['label'].mean():.2f}")

    urls = df['url'].tolist()
    labels = df['label'].tolist()

    train_urls, val_urls, train_labels, val_labels = train_test_split(
        urls, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"Loading Tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Tokenizing datasets...")
    train_encodings = tokenizer(train_urls, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_urls, truncation=True, padding=True, max_length=128)

    train_dataset = URLDataset(train_encodings, train_labels)
    val_dataset = URLDataset(val_encodings, val_labels)

    logger.info(f"Loading Model ({model_name})...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=args.epochs,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,            
        compute_metrics=compute_metrics
    )

    logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        logger.info("No checkpoint found, starting from scratch...")
        trainer.train()

    logger.info(f"Saving fine-tuned model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="ml_pipeline/data/PhiUSIIL_Phishing_URL_Dataset.csv")
    parser.add_argument("--sample_size", type=int, default=20000, help="Number of rows to sample (0 = all)")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)
