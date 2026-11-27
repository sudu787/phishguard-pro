# Full training pipeline, in order. Each step needs the previous one's output.
#   .\run_massive_training.ps1 -Python C:\path\to\venv_gpu\Scripts\python.exe
# Optional first steps (network): python ml_pipeline/fetch_commoncrawl_benign.py   (~0.7 GB streamed)
#                                 python ml_pipeline/fetch_phreshphish.py          (~0.1 GB, needs pyarrow)
param([string]$Python = ".\venv_gpu\Scripts\python.exe")
$ErrorActionPreference = "Stop"

function Invoke-Step($label, [string[]]$arguments) {
    Write-Host "== $label"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "$label failed (exit $LASTEXITCODE)" }
}

Invoke-Step "Combine datasets"       @("ml_pipeline/combine_datasets.py")
Invoke-Step "Extract URL features"   @("ml_pipeline/extract_bulk_features.py")
Invoke-Step "Train XGBoost + text"   @("-m", "ml_pipeline.train", "--data", "ml_pipeline/data/megaset_features.csv", "--output", "ml_pipeline/models", "--device", "cuda")
Invoke-Step "Fine-tune BERT"         @("-m", "ml_pipeline.finetune_bert", "--data", "ml_pipeline/data/megaset_features.csv")
Invoke-Step "Blend XGBoost + BERT"   @("-m", "ml_pipeline.blend")
Write-Host "Done. See ml_pipeline/models/training_report.json and blend.json."
