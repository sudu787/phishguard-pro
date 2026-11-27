"""
PhreshPhish page HTML → page features for the Tier 2 (URL + page) model.
  ml_pipeline/data/phreshphish/<out>/{split}-NNNNN.csv (out = argv[1], default pages): sha256, url, label, DOM_FEATURE_COLUMNS, page_text

Streams the parquet html column and keeps only the features + PAGE_TEXT_CHARS of title/visible text;
raw HTML never touches disk. dom_features() mirrors extension/content/content.js so training sees
what the extension sends. Resumable: finished files are skipped.

Train: every TRAIN_STRIDE-th file (1 = all 27 GB; 4 ≈ 7 GB). Test: only benchmark rows (benchmark scoring, never training).
Needs pyarrow + bs4 + lxml (training-time only):  python ml_pipeline/fetch_phreshphish_html.py
"""
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The pyarrow Python may lack tldextract; only the column lists are needed from features.
sys.modules.setdefault("tldextract", __import__("types").SimpleNamespace(TLDExtract=lambda *a, **k: None))
from ml_pipeline.features import DOM_FEATURE_COLUMNS, PAGE_TEXT_CHARS  # noqa: E402

DIR = Path(__file__).parent / "data" / "phreshphish"
REPO = "phreshphish/phreshphish"
TRAIN_STRIDE = 1
MAX_PARSE = 1_000_000  # chars of HTML parsed per page (bounds time on huge pages)

SUSPICIOUS_INPUT = re.compile(r"passwd|password|pass|pwd|login|user|username|email|credit|card|ccnum|cvv|ssn|social|pin|otp|mfa", re.I)
TEXT_URL = re.compile(r"https?://[^\s\"'<>]+", re.I)
OBFUSCATION = [re.compile(p) for p in (r"\beval\s*\(", r"\bunescape\s*\(", r"String\.fromCharCode", r"\\x[0-9a-fA-F]{2}",
                                       r"\batob\s*\(", r"\bbtoa\s*\(", r"document\.write\s*\(", r"\\u[0-9a-fA-F]{4}",
                                       r"Function\s*\(\s*['\"`]")]
COPYRIGHT = re.compile(r"©|copyright|\bcopr\b|\(c\)\s*\d{4}", re.I)
LOGIN_WORDS = re.compile(r"sign\s*in|log\s*in|login|verify your|confirm your|secure your|your account|your credential", re.I)


def _host(href: str, base: str) -> str | None:
    try:
        return urlparse(urljoin(base, href)).hostname
    except ValueError:
        return None


def _depth(node, d: int = 0) -> int:  # content.js _getDOMDepth: capped at 50, first 100 children
    if node is None or d > 50:
        return d
    kids = [c for c in node.children if getattr(c, "name", None)][:100]
    # Browsers wrap a table's bare <tr> rows in an implied <tbody> (one level deeper); lxml doesn't.
    step = lambda c: 2 if node.name == "table" and c.name == "tr" else 1
    return max([d] + [_depth(c, d + step(c)) for c in kids])


BLOCK_TAGS = ["address", "article", "aside", "blockquote", "br", "dd", "details", "dialog", "div", "dl", "dt",
              "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
              "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table", "tr", "ul", "td", "th"]


def _inner_text(node) -> str:
    """≈ element.innerText, whitespace-collapsed as content.js does: inline elements join without a
    separator ("Hacker Newsnew"), block elements and table cells break. Mutates `node`."""
    for t in node.find_all(BLOCK_TAGS):
        t.insert_before("\n")
        t.insert_after("\n")
    return " ".join(node.get_text("").split())


def _px(v) -> int | None:
    m = re.match(r"\s*(\d+)", str(v or ""))
    return int(m.group(1)) if m else None


def dom_features(html: str, url: str) -> dict:
    """Python port of content.js extractDOMFeatures() (+ page_text). Browser-only bits are approximated:
    iframe size/visibility from attributes and inline style (browser default 300x150), innerText ≈ _inner_text()."""
    s = BeautifulSoup(html[:MAX_PARSE], "lxml")
    host = urlparse(url if "://" in url else "http://" + url).hostname
    f = dict.fromkeys(DOM_FEATURE_COLUMNS, 0)

    forms = s.find_all("form")
    f["num_forms"] = len(forms)
    for form in forms:
        action = (form.get("action") or "").strip().lower()
        if not action or action in ("#", "javascript:void(0)", "about:blank"):
            f["has_empty_form_action"] = 1
        elif action.startswith("javascript:"):
            f["has_javascript_form_action"] = 1
        else:
            h = _host(action, url)
            if h is None:
                f["has_empty_form_action"] = 1
            elif h != host:
                f["has_external_form_action"] = 1

    inputs = s.find_all("input")
    f["num_password_inputs"] = sum((i.get("type") or "").lower() == "password" for i in inputs)
    f["num_hidden_inputs"] = sum((i.get("type") or "").lower() == "hidden" for i in inputs)
    f["num_suspicious_inputs"] = sum(bool(SUSPICIOUS_INPUT.search(i.get("name") or i.get("id") or i.get("placeholder") or ""))
                                     for i in inputs)

    links = s.find_all("a", href=True)
    f["num_links"] = len(links)
    for a in links:
        href = a["href"].strip()
        if not href or href in ("#", "javascript:void(0)", "javascript:;"):
            f["num_null_links"] += 1
            continue
        h = _host(href, url)
        if h is None:
            continue
        f["num_external_links"] += h != host
        for t in TEXT_URL.findall(a.get_text()):
            th = _host(t, url)
            if th and th != h:
                f["num_deceptive_links"] += 1
                break
    if f["num_links"]:
        f["pct_external_links"] = round(f["num_external_links"] / f["num_links"] * 100)

    scripts = s.find_all("script")
    f["num_scripts"] = len(scripts)
    for sc in scripts:
        if sc.get("src"):
            h = _host(sc["src"], url)
            f["num_external_scripts"] += bool(h and h != host)
        elif (txt := sc.string or "") and any(p.search(txt) for p in OBFUSCATION):
            f["has_obfuscated_scripts"] = 1

    frames = s.find_all("iframe")
    f["num_iframes"] = len(frames)
    for fr in frames:
        st = (fr.get("style") or "").replace(" ", "").lower()
        css = lambda prop: (m := re.search(rf"(?<![-\w]){prop}:(\d+)", st)) and int(m.group(1))
        w = _px(fr.get("width")) or css("width") or 300  # parseInt('0') is 0 → falls to default, as in JS
        h = _px(fr.get("height")) or css("height") or 150
        if w <= 1 or h <= 1 or "display:none" in st or "visibility:hidden" in st or re.search(r"opacity:0(?:\.0*)?(?:;|$)", st):
            f["num_invisible_iframes"] += 1

    for link in s.find_all("link", href=True):
        if "icon" in " ".join(link.get("rel") or []).lower():
            h = _host(link["href"], url)
            if h and h != host:
                f["favicon_from_external_domain"] = 1

    f["has_meta_refresh"] = int(any((m.get("http-equiv") or "").lower() == "refresh" for m in s.find_all("meta")))
    body = s.body
    f["disables_right_click"] = int(bool(body and re.search(r"return\s*false", body.get("oncontextmenu") or "", re.I)))
    title = " ".join((s.title.get_text() if s.title else "").split())  # document.title collapses whitespace
    f["has_empty_title"] = int(not title)
    f["has_status_bar_customization"] = int(any(re.search(r"window\.status", el.get("onmouseover") or "", re.I)
                                                for el in s.find_all(attrs={"onmouseover": True})))
    f["page_size_bytes"] = len(html)
    f["dom_depth"] = _depth(body)

    for t in s(["script", "style", "noscript", "template"]):
        t.decompose()
    text = _inner_text(body or s)
    low = text[:10_000].lower()
    f["has_copyright"] = int(bool(COPYRIGHT.search(low)))
    f["has_login_keywords"] = int(bool(LOGIN_WORDS.search(low)))
    f["page_text"] = f"{title} | {text}"[:PAGE_TEXT_CHARS]
    return f


def _row(args):
    sha, url, label, html = args
    try:
        return {"sha256": sha, "url": url, "label": label, **dom_features(html or "", url)}
    except Exception:  # a malformed page must not kill a multi-hour run
        return None


def main(out_name: str = "pages"):
    out_dir = DIR / out_name
    import pyarrow.parquet as pq  # streaming only: dom_features() is importable without them
    from huggingface_hub import HfFileSystem

    out_dir.mkdir(parents=True, exist_ok=True)
    B = DIR / "benchmark"
    bench = set((B / "benign.txt").read_text().split()) | set((B / "phish.txt").read_text().split())
    fs = HfFileSystem()
    with Pool(max(1, __import__("os").cpu_count() - 2)) as pool:
        for split in ["train", "test"]:
            paths = sorted(fs.glob(f"datasets/{REPO}/data/{split}-*.parquet"))
            for path in (paths[::TRAIN_STRIDE] if split == "train" else paths):
                out = out_dir / f"{split}-{Path(path).stem.split('-')[1]}.csv"
                if out.exists():
                    continue
                t0, parts = time.time(), []
                with fs.open(path, block_size=1 << 22) as fh:
                    pf = pq.ParquetFile(fh)
                    for rg in range(pf.num_row_groups):  # one row group in memory at a time
                        d = pf.read_row_group(rg, columns=["sha256", "url", "label", "html"]).to_pandas()
                        if split == "test":
                            d = d[d.sha256.isin(bench)]
                        rows = pool.map(_row, zip(d.sha256, d.url, d.label, d.html), chunksize=32)
                        parts.append(pd.DataFrame([r for r in rows if r]))
                df = pd.concat(parts, ignore_index=True)
                df.to_csv(out.with_suffix(".tmp"), index=False)
                out.with_suffix(".tmp").replace(out)
                print(f"{out.name}: {len(df):,} pages ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])  # optional output folder name under data/phreshphish/ (default pages)
