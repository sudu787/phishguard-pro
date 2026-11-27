"""
Sample legitimate deep links from the Common Crawl URL index → ml_pipeline/data/commoncrawl_benign.csv

Why: the benign side of the training data is mostly top-site homepages and old
directory listings, so ordinary small sites with real paths look unusual to the
model. Common Crawl captures are what normal browsing looks like.

Streams the head of evenly spaced index shards (no full download), keeps HTML
pages that returned 200, drops user-content hosts and any domain seen as
phishing in megaset.csv, and keeps at most PER_DOMAIN random URLs per site.

ponytail: Common Crawl also crawls some phishing pages; this accepts that small
label noise rather than cross-checking every URL against a live feed.

Run from the repo root: python ml_pipeline/fetch_commoncrawl_benign.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import gzip
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from ml_pipeline.features import USER_CONTENT_HOSTS, normalise, registered_domain, url_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_SHARDS, TOTAL_SHARDS = 24, 300
LINES_PER_SHARD = 300_000
PER_DOMAIN = 3
OUT = Path("ml_pipeline/data/commoncrawl_benign.csv")


def read_shard(crawl: str, i: int) -> list[tuple[str, str]]:
    url = f"https://data.commoncrawl.org/cc-index/collections/{crawl}/indexes/cdx-{i:05d}.gz"
    out = []
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with gzip.open(r.raw, "rt", encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f):
                if n >= LINES_PER_SHARD:
                    break
                try:
                    rec = json.loads(line.split(" ", 2)[2])
                except (IndexError, ValueError):
                    continue
                u = rec.get("url", "")
                if (rec.get("status") != "200" or u.endswith("robots.txt")
                        or rec.get("mime-detected", rec.get("mime")) != "text/html"):
                    continue
                host = normalise(u)[1]
                if any(host == h or host.endswith("." + h) for h in USER_CONTENT_HOSTS):
                    continue
                out.append((registered_domain(host), u))
    logger.info(f"shard {i:05d}: {len(out):,} candidate URLs")
    return out


def main():
    crawl = requests.get("https://index.commoncrawl.org/collinfo.json", timeout=60).json()[0]["id"]
    shards = sorted({round(k * (TOTAL_SHARDS - 1) / (N_SHARDS - 1)) for k in range(N_SHARDS)})
    logger.info(f"Crawl {crawl}: reading {LINES_PER_SHARD:,} lines from {len(shards)} shards")

    mega = pd.read_csv("ml_pipeline/data/megaset.csv", usecols=["url", "label"])
    phish_domains = set(mega.loc[mega["label"] == 1, "url"].astype(str).map(registered_domain))
    logger.info(f"Excluding {len(phish_domains):,} domains labelled phishing in megaset.csv")

    by_domain = defaultdict(dict)  # domain → {normalised text: url}
    with ThreadPoolExecutor(6) as pool:
        for rows in pool.map(lambda i: read_shard(crawl, i), shards):
            for dom, u in rows:
                if dom not in phish_domains:
                    by_domain[dom].setdefault(url_text(u), u)

    rng = random.Random(42)
    urls = [u for d in by_domain.values() for u in rng.sample(list(d.values()), min(PER_DOMAIN, len(d)))]
    pd.DataFrame({"url": urls, "type": "benign"}).to_csv(OUT, index=False)
    logger.info(f"Saved {len(urls):,} URLs from {len(by_domain):,} domains to {OUT}")


if __name__ == "__main__":
    main()
