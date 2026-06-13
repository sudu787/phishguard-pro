from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "ml_pipeline/models/phishbert"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

urls = [
    "http://example.com/login",
    "https://google.com",
    "http://secure-update-paypal-account.com/login.php",
    "https://www.uni-mainz.de",
    "https://www.southbankmosaics.com"
]

for url in urls:
    inputs = tokenizer(url, return_tensors="pt", truncation=True, max_length=128, padding=True)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    print(f"URL: {url}")
    print(f"Inputs: {inputs}")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        print(f"URL: {url}")
        print(f"Logits: {outputs.logits}")
        print(f"Probs: {probs}")
        print("-" * 30)
