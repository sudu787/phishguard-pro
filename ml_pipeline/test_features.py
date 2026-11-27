"""Self-check for the URL features and threshold picking. Run: python -m ml_pipeline.test_features"""
import numpy as np
from sklearn.metrics import f1_score

from ml_pipeline.features import (
    URL_FEATURE_COLUMNS, URLFeatureExtractor, abused_hosts, abused_platforms, normalise, registered_domain, site_group,
    url_text, url_tokens,
)
from ml_pipeline.train import make_text_model, pick_thresholds

e = URLFeatureExtractor()

# 1. How a dataset formats a URL must not change its features (the old shortcut).
same = ["en.wikipedia.org/wiki/Phishing", "https://en.wikipedia.org/wiki/Phishing",
        "http://www.en.wikipedia.org/wiki/Phishing/"]
first = e.extract(same[0])
for u in same[1:]:
    assert e.extract(u) == first, u
assert first["shannon_entropy"] > 0 and first["host_length"] == len("en.wikipedia.org")
assert list(first) == URL_FEATURE_COLUMNS

# 1b. Malformed URLs from real feeds don't crash (urlparse raises on an unbalanced "[").
assert normalise("http://evil[.com/login?a=1") == ("evil[.com/login?a=1", "evil[.com", "/login")
assert normalise("https://[::1]/x")[1] == "::1"  # valid IPv6 still goes through urlparse

# 2. Shortener match is on the registered domain, not a substring.
assert e.extract("https://microsoft.com")["tiny_url"] == 0
assert e.extract("https://t.co/abc")["tiny_url"] == 1

# 3. Brand outside the real domain.
assert e.extract("http://paypal.com.secure-login.xyz/verify")["brand_outside_domain"] == 1
assert e.extract("https://www.paypal.com/signin")["brand_outside_domain"] == 0
assert e.extract("https://google.com/search?q=paypal")["brand_outside_domain"] == 0

# 3b. Newer lexical features.
f = e.extract("http://paypa1-secure.web.app:8080/login/verify.php?a=1&b=2")
assert (f["free_host"], f["has_port"], f["script_ext"], f["num_params"]) == (1, 1, 1, 2), f
assert f["num_dots"] == 2 and f["num_hyphens"] == 1 and f["host_digits"] == 1
assert e.extract("http://paypa1-secure.com/x")["brand_typo"] == 1     # leet
assert e.extract("http://rnicrosoft-support.net/")["brand_typo"] == 1  # rn → m
assert e.extract("https://www.paypal.com/signin")["brand_typo"] == 0   # the real brand
assert e.extract("https://news.bbc.co.uk/")["brand_typo"] == e.extract("https://example.com")["free_host"] == 0
assert e.extract("https://a.com/abcdefghij-x")["longest_token"] == 10

# 3c. Word tokens for the text model: host tokens tagged, bigrams included.
assert url_tokens("paypal.com.evil.xyz/paypal/login") == [
    "h:paypal", "h:com", "h:evil", "h:xyz", "paypal", "login",
    "h:paypal h:com", "h:com h:evil", "h:evil h:xyz", "h:xyz paypal", "paypal login"]

# 4. Registered domain: public-suffix aware, "www." handled as a prefix only.
assert registered_domain("https://a.b.bbc.co.uk/x") == "bbc.co.uk"
assert registered_domain("web.de") == "web.de"
assert registered_domain("http://1.2.3.4/login") == "1.2.3.4"
assert registered_domain("https://evil.github.io/x") == "github.io"
assert registered_domain("https://evil.github.io/x", private=True) == "evil.github.io"

# Abused platforms are learnt from labels; grouping splits them per host.
urls = [f"https://p{i}.gitbook.io/x" for i in range(20)] + ["https://a.example.com/1", "https://b.example.com/2"]
plat = abused_platforms(urls, [1] * 22)
assert plat == {"gitbook.io"}, plat                                # 20 phishing hosts → platform; 2 → not
assert abused_platforms(urls[:19], [1] * 19) == set()
pages = [f"https://new.express.adobe.com/webpage/{i}" for i in range(40)]
assert abused_hosts(pages, [1] * 20 + [0] * 20) == {"new.express.adobe.com"}   # 20 phishing, 50% → host
assert abused_hosts(pages, [1] * 19 + [0] * 21) == set()                      # 19 phishing → not
assert abused_hosts(pages + [f"https://new.express.adobe.com/b/{i}" for i in range(5)],
                    [1] * 20 + [0] * 25) == set()                             # 20 of 45 < 50% → not
assert site_group("https://p1.gitbook.io/x", plat) == "p1.gitbook.io"
assert site_group("https://a.example.com/1", plat) == "example.com"   # ordinary site: one group
assert site_group("https://x.vercel.app/", set()) == "x.vercel.app"   # user-content host: per host
assert site_group("https://docs.google.com/forms/d/e/1FAIpQLSabc123/viewform?usp=sf_link", set()) == \
    site_group("https://docs.google.com/forms/d/e/1FAIpQLSabc123/formResponse", set()) != \
    site_group("https://docs.google.com/forms/d/e/1FAIpQLSxyz789/viewform", set())  # per document

# 4b. The text model sees the same string however the dataset wrote the URL.
assert len({url_text(u) for u in same}) == 1

# 4c. The text model learns character patterns it was never given as features.
legit = [f"https://{w}.com/{p}" for w in ("news", "shop", "maps", "mail", "blog", "docs", "wiki", "music")
         for p in ("home", "about", "help", "search", "contact")]
phish = [f"http://{w}-paypa1-verify.top/{p}" for w in ("acct", "secure", "auth", "user", "id", "pay", "web", "app")
         for p in ("signin", "confirm", "update", "unlock", "billing")]
tm = make_text_model().fit([url_text(u) for u in legit + phish], [0] * len(legit) + [1] * len(phish))
p_bad, p_good = tm.predict_proba([url_text("http://login-paypa1-verify.top/restore"),
                                  url_text("https://travel.com/about")])[:, 1]
assert p_bad > 0.5 > p_good, (p_bad, p_good)

# 5. Thresholds hit the target false-positive rate on the data they're fitted on.
rng = np.random.default_rng(0)
y = np.r_[np.zeros(100_000), np.ones(1_000)]
p = np.r_[rng.beta(2, 8, 100_000), rng.beta(8, 2, 1_000)]
thr = pick_thresholds(y, p, 0.001, 0.01)
assert abs((p[y == 0] >= thr["phishing"]).mean() - 0.001) < 2e-4
assert abs((p[y == 0] >= thr["suspicious"]).mean() - 0.01) < 1e-3
assert thr["suspicious"] <= thr["phishing"]
f1_at = lambda t: f1_score(y, p >= t)
assert f1_at(thr["balanced"]) >= max(f1_at(thr["phishing"]), f1_at(thr["suspicious"]))  # max-F1 point

# 6. The page-feature port matches what content.js saw in a real browser (news.ycombinator.com, 2026-11-27):
#    innerText joins inline elements, and rows of a bare <table> sit one level deeper (implied <tbody>).
try:
    from ml_pipeline.fetch_phreshphish_html import dom_features
except ImportError:  # bs4/lxml are training-time only
    print("skipped page-feature checks (bs4/lxml not installed)")
else:
    page = ('<html><head><title> Hacker\n News </title></head><body><table><tr><td><a href="/news">Hacker News</a>'
            '<a href="/newest">new</a> | <a href="/front">past</a></td></tr></table>'
            '<p>AMD patches (<a href="https://devnull.land/x">devnull.land</a>)</p></body></html>')
    f = dom_features(page, "https://news.ycombinator.com/")
    assert f["page_text"] == "Hacker News | Hacker Newsnew | past AMD patches (devnull.land)", f["page_text"]
    assert f["dom_depth"] == 5, f["dom_depth"]  # body>table>(tbody)>tr>td>a
    assert (f["num_links"], f["num_external_links"], f["pct_external_links"]) == (4, 1, 25)

print("all feature/threshold checks passed")
