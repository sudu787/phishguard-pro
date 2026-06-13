.\venv_gpu\Scripts\python.exe -m pip install pandas numpy scikit-learn transformers accelerate
.\venv_gpu\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.\venv_gpu\Scripts\python.exe ml_pipeline/finetune_bert.py --sample_size 50000 --epochs 1
