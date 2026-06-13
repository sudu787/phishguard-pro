# Launch both the XGBoost and BERT training in parallel using the CUDA environment

Write-Host "Starting Deep Learning BERT Training in the background..."
Start-Process -FilePath ".\venv_gpu\Scripts\python.exe" -ArgumentList "ml_pipeline/finetune_bert.py --data_path ml_pipeline/data/megaset.csv --sample_size 0 --epochs 3" -NoNewWindow

Write-Host "Starting XGBoost Massive Training in the background..."
Start-Process -FilePath ".\venv_gpu\Scripts\python.exe" -ArgumentList "-m ml_pipeline.train --data ml_pipeline/data/megaset_features.csv --output ml_pipeline/models --trials 10 --cv-folds 10" -NoNewWindow

Write-Host "Both processes started!"
