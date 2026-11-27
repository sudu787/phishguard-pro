"""Self-check for the top-sites allowlist. Run from backend/: python verify_predictor.py"""
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for ml_pipeline

from app.services.predictor import load_top_domains, top_site

with tempfile.TemporaryDirectory() as d:
    z = Path(d) / "tranco.csv.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("top-1m.csv", "1,google.com\n2,paypal.com\n3,github.io\n4,bit.ly\n5,web.de\n")
    top = load_top_domains(str(z), top_n=4)

assert top == {"google.com", "paypal.com", "github.io", "bit.ly"}  # top_n respected
assert load_top_domains("does/not/exist.zip", 10) == set()           # missing list → disabled

# Real top sites are allowlisted, however the URL is written.
assert top_site("https://www.paypal.com/signin", top) == "paypal.com"
assert top_site("paypal.com/signin", top) == "paypal.com"
assert top_site("https://mail.google.com/x", top) == "google.com"

# Look-alikes and tricks are not.
for url in [
    "http://paypal.com.secure-login.xyz/verify",   # brand as subdomain
    "http://paypal.com@evil.xyz/login",             # userinfo trick: real host is evil.xyz
    "http://xn--pypal-4ve.com/login",               # punycode look-alike
    "https://evil.github.io/paypal",                # user site on a top hosting domain
    "https://docs.google.com/forms/d/e/abc/viewform",  # user content on a top domain
    "https://bit.ly/3abc",                          # shortener: can point anywhere
    "http://1.2.3.4/paypal",
]:
    assert top_site(url, top) is None, url

assert top_site("https://www.paypal.com", set()) is None  # empty list → no-op

# Abused platforms (train.py's platforms.json): the site itself yes, anyone's subdomain no.
plat = {"gitbook.io"}
assert top_site("https://evil-wallet.gitbook.io/login", top | plat, plat) is None
assert top_site("https://www.gitbook.io/pricing", top | plat, plat) == "gitbook.io"
assert top_site("https://mail.google.com/x", top, plat) == "google.com"  # not a platform → unchanged
# Hosts mostly phishing in our labels (platforms.json "hosts") are never allowlisted; the rest of the site is.
top_a = top | {"adobe.com"}
assert top_site("https://new.express.adobe.com/webpage/x", top_a, plat, {"new.express.adobe.com"}) is None
assert top_site("https://www.adobe.com/acrobat", top_a, plat, {"new.express.adobe.com"}) == "adobe.com"
print("allowlist checks passed")
