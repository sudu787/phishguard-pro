"""
PhreshPhish (CC-BY-4.0, 2024-25 phishing + benign pages) → ml_pipeline/data/phreshphish/{train,test}.csv + benchmark/

Reads only the url/label/target/date columns of the parquet files (column-projected
range reads, ~100 MB instead of the 36 GB with HTML). combine_datasets.py trains on
train.csv; test.csv + benchmark/ are the external benchmark and must never be trained on.

Needs pyarrow + huggingface_hub (not in requirements.txt: training-time only):
  python ml_pipeline/fetch_phreshphish.py
"""
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, snapshot_download

OUT = Path(__file__).parent / "data" / "phreshphish"
REPO = "phreshphish/phreshphish"
COLS = ["sha256", "url", "label", "target", "date"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO, repo_type="dataset", allow_patterns=["benchmark/*"], local_dir=OUT)
    fs = HfFileSystem()
    for split in ["train", "test"]:
        t0, frames = time.time(), []
        for path in sorted(fs.glob(f"datasets/{REPO}/data/{split}-*.parquet")):
            with fs.open(path, block_size=1 << 20) as f:  # seekable; only the COLS chunks are fetched
                frames.append(pq.ParquetFile(f).read(columns=COLS).to_pandas())
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(OUT / f"{split}.csv", index=False)
        print(f"{split}: {len(df):,} rows {df.label.value_counts().to_dict()} "
              f"{df.date.min()}..{df.date.max()} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
