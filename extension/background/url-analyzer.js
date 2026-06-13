/**
 * PhishGuard V2 — Local URL Feature Extraction & Risk Scoring
 * background/url-analyzer.js
 *
 * Performs 20+ heuristic checks on URL structure to detect phishing patterns.
 * Runs entirely client-side with zero network requests.
 *
 * Architecture:
 *   1. Parse the URL into components (scheme, host, path, query, etc.)
 *   2. Run each feature detector — returns { triggered, weight, feature }
 *   3. Sum triggered weights → apply sigmoid normalisation → score in [0, 1]
 *   4. Return score + verdict + ranked feature list for explainability
 *
 * Designed to be replaceable with ONNX ML model inference in Phase 1b.
 */

// ─── Known URL shortener domains ──────────────────────────────────────────────
const SHORTENER_DOMAINS = new Set([
  'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd',
  'buff.ly', 'short.to', 'tr.im', 'tiny.cc', 'cutt.ly', 'rb.gy',
  'shorturl.at', 'adf.ly', 'bit.do', 'lnkd.in', 'db.tt', 'qr.ae',
  'yourls.org', 'v.gd', 'link.zip.net', 'rebrand.ly', 'tny.im',
  'surl.li', 'shortcm.li', 'bl.ink', 'clck.ru',
]);

// ─── TLDs commonly abused in phishing ─────────────────────────────────────────
const SUSPICIOUS_TLDS = new Set([
  '.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.xyz', '.buzz',
  '.club', '.work', '.click', '.link', '.info', '.support',
  '.review', '.stream', '.racing', '.win', '.bid', '.download',
  '.loan', '.men', '.party', '.science', '.trade', '.gdn',
  '.accountant', '.cricket', '.date', '.faith', '.rocks',
]);

// ─── Brands commonly targeted by phishing ─────────────────────────────────────
const BRAND_TARGETS = [
  { brand: 'PayPal',    legit: 'paypal.com',        variants: ['paypa1', 'paypaI', 'paypol', 'pay-pal', 'peypal', 'payp4l', 'paypai', 'paypal-login', 'paypal-verify'] },
  { brand: 'Google',    legit: 'google.com',         variants: ['g00gle', 'googIe', 'gooogle', 'go0gle', 'g0ogle', 'google-login'] },
  { brand: 'Microsoft', legit: 'microsoft.com',      variants: ['micr0soft', 'mlcrosoft', 'microsft', 'microsoft-login', 'microsoftonline-verify'] },
  { brand: 'Apple',     legit: 'apple.com',          variants: ['app1e', 'appIe', 'apple-id', 'apple-verify', 'icloud-login'] },
  { brand: 'Amazon',    legit: 'amazon.com',         variants: ['amaz0n', 'arnazon', 'amazom', 'amazon-login', 'amazn'] },
  { brand: 'Netflix',   legit: 'netflix.com',        variants: ['netf1ix', 'netfIix', 'netflix-update', 'netfliix'] },
  { brand: 'Facebook',  legit: 'facebook.com',       variants: ['faceb00k', 'faceboak', 'facebook-login', 'fb-login'] },
  { brand: 'Instagram', legit: 'instagram.com',      variants: ['lnstagram', 'instagrom', 'instagram-verify'] },
  { brand: 'LinkedIn',  legit: 'linkedin.com',       variants: ['linkedln', 'linkdin', 'linkedin-verify'] },
  { brand: 'Chase',     legit: 'chase.com',          variants: ['chas3', 'chase-secure', 'chase-verify', 'chase-login'] },
  { brand: 'Wells Fargo', legit: 'wellsfargo.com',   variants: ['wellsfarg0', 'wells-fargo', 'wellsfargo-login'] },
  { brand: 'Bank of America', legit: 'bankofamerica.com', variants: ['bankofarnerica', 'bank0famerica'] },
  { brand: 'HDFC',      legit: 'hdfcbank.com',       variants: ['hdfc-netbanking', 'hdfcbank-login', 'hdfc-secure'] },
  { brand: 'SBI',       legit: 'onlinesbi.com',      variants: ['sbi-online', 'sbi-netbanking', 'onlinesbi-login'] },
  { brand: 'WhatsApp',  legit: 'whatsapp.com',       variants: ['whatsap', 'whats-app', 'whatsapp-verify'] },
  { brand: 'Dropbox',   legit: 'dropbox.com',        variants: ['dr0pbox', 'dropb0x', 'dropbox-login'] },
  { brand: 'Twitter',   legit: 'twitter.com',        variants: ['twltter', 'tw1tter', 'twitter-verify'] },
  { brand: 'GitHub',    legit: 'github.com',         variants: ['githvb', 'guthub', 'github-login'] },
];

// ─── Keywords commonly found in phishing URLs ─────────────────────────────────
const PHISHING_KEYWORDS = [
  'login', 'signin', 'sign-in', 'log-in', 'verify', 'verification',
  'confirm', 'update', 'secure', 'account', 'banking', 'password',
  'credential', 'suspend', 'restrict', 'unlock', 'validate', 'wallet',
  'authenticate', 'identity', 'recover', 'reactivate', 'unusual',
  'alert', 'notification', 'expire', 'limited', 'urgent',
];

// ─── Suspicious port numbers ──────────────────────────────────────────────────
const SUSPICIOUS_PORTS = new Set([
  '8080', '8443', '8888', '3000', '4443', '9090', '1337', '4444',
  '5555', '6666', '7777', '8000', '8081', '8082', '9000', '9999',
]);


// ═══════════════════════════════════════════════════════════════════════════════
//  URL Analyzer
// ═══════════════════════════════════════════════════════════════════════════════

export class URLAnalyzer {

  /**
   * Analyse a URL and return a phishing risk assessment.
   *
   * @param  {string} url — The full URL (e.g. "https://example.com/path")
   * @return {{ score: number, verdict: string, features: Array, rawScore: number, url: string }}
   */
  analyze(url) {
    let parsed;
    try {
      parsed = new URL(url);
    } catch {
      // Unparseable URL is itself suspicious
      return {
        score: 0.85,
        verdict: 'phishing',
        features: [{ name: 'unparseable_url', label: 'URL could not be parsed', severity: 'critical', weight: 5.0 }],
        rawScore: 5.0,
        url,
      };
    }

    const hostname = parsed.hostname.toLowerCase();
    const fullUrl  = url.toLowerCase();
    const path     = parsed.pathname || '';
    const search   = parsed.search || '';

    const triggered = [];

    // ── 1. IP address as host ─────────────────────────────────────────────
    if (this._isIPAddress(hostname)) {
      triggered.push({
        name: 'ip_address_host',
        label: 'IP address used instead of domain name',
        severity: 'critical',
        weight: 3.5,
      });
    }

    // ── 2. @ symbol in URL (credential injection) ─────────────────────────
    if (fullUrl.includes('@')) {
      triggered.push({
        name: 'at_sign',
        label: '@ symbol in URL — may hide the real destination',
        severity: 'high',
        weight: 2.5,
      });
    }

    // ── 3. URL length ─────────────────────────────────────────────────────
    if (url.length >= 100) {
      triggered.push({
        name: 'very_long_url',
        label: `Unusually long URL (${url.length} characters)`,
        severity: 'medium',
        weight: 1.5,
      });
    } else if (url.length >= 75) {
      triggered.push({
        name: 'long_url',
        label: `Long URL (${url.length} characters)`,
        severity: 'low',
        weight: 0.8,
      });
    }

    // ── 4. Double-slash redirect in path ──────────────────────────────────
    const lastDoubleSlash = fullUrl.lastIndexOf('//');
    if (lastDoubleSlash > 7) {
      triggered.push({
        name: 'double_slash_redirect',
        label: 'Double-slash (//) redirect detected in URL path',
        severity: 'medium',
        weight: 1.5,
      });
    }

    // ── 5. Hyphen in domain (prefix-suffix attack) ────────────────────────
    const domainWithoutTld = hostname.split('.').slice(0, -1).join('.');
    const hyphenCount = (domainWithoutTld.match(/-/g) || []).length;
    if (hyphenCount >= 3) {
      triggered.push({
        name: 'many_hyphens',
        label: `Domain contains ${hyphenCount} hyphens — common in phishing`,
        severity: 'high',
        weight: 2.5,
      });
    } else if (hyphenCount >= 1) {
      triggered.push({
        name: 'prefix_suffix',
        label: 'Hyphen in domain name — possible prefix-suffix attack',
        severity: 'medium',
        weight: 1.2,
      });
    }

    // ── 6. Excessive subdomains ───────────────────────────────────────────
    const cleanHost = hostname.replace(/^www\./, '');
    const parts = cleanHost.split('.');
    const subdomainCount = parts.length - 2; // subtract SLD + TLD
    if (subdomainCount >= 3) {
      triggered.push({
        name: 'many_subdomains',
        label: `${subdomainCount} subdomains — used to obscure the real domain`,
        severity: 'high',
        weight: 2.0,
      });
    } else if (subdomainCount === 2) {
      triggered.push({
        name: 'extra_subdomain',
        label: 'Extra subdomain level detected',
        severity: 'low',
        weight: 0.7,
      });
    }

    // ── 7. HTTP (no TLS) ──────────────────────────────────────────────────
    if (parsed.protocol === 'http:') {
      triggered.push({
        name: 'no_https',
        label: 'Site uses HTTP — no encryption',
        severity: 'medium',
        weight: 1.2,
      });
    }

    // ── 8. "https" word in domain name (deceptive) ────────────────────────
    if (hostname.includes('https') || hostname.includes('secure')) {
      triggered.push({
        name: 'https_in_domain',
        label: 'Security keywords ("https"/"secure") used in domain name',
        severity: 'high',
        weight: 2.0,
      });
    }

    // ── 9. URL shortener ──────────────────────────────────────────────────
    if (SHORTENER_DOMAINS.has(hostname) || SHORTENER_DOMAINS.has(cleanHost)) {
      triggered.push({
        name: 'url_shortener',
        label: 'URL shortener service — hides the real destination',
        severity: 'medium',
        weight: 1.5,
      });
    }

    // ── 10. Suspicious TLD ────────────────────────────────────────────────
    const tld = '.' + parts[parts.length - 1];
    if (SUSPICIOUS_TLDS.has(tld)) {
      triggered.push({
        name: 'suspicious_tld',
        label: `Suspicious top-level domain "${tld}" — commonly abused`,
        severity: 'medium',
        weight: 1.5,
      });
    }

    // ── 11. Phishing keywords in URL ──────────────────────────────────────
    const urlTokens = (path + search).toLowerCase();
    const matchedKeywords = PHISHING_KEYWORDS.filter(kw => urlTokens.includes(kw));
    if (matchedKeywords.length >= 3) {
      triggered.push({
        name: 'many_phishing_keywords',
        label: `Multiple phishing keywords: ${matchedKeywords.slice(0, 3).join(', ')}`,
        severity: 'high',
        weight: 2.0,
      });
    } else if (matchedKeywords.length >= 1) {
      triggered.push({
        name: 'phishing_keywords',
        label: `Phishing keyword in URL: "${matchedKeywords[0]}"`,
        severity: 'low',
        weight: 0.8,
      });
    }

    // ── 12. Brand typosquatting ───────────────────────────────────────────
    const typosquat = this._checkTyposquatting(hostname);
    if (typosquat) {
      triggered.push({
        name: 'brand_typosquatting',
        label: `Possible ${typosquat.brand} impersonation — domain mimics "${typosquat.legit}"`,
        severity: 'critical',
        weight: 3.5,
      });
    }

    // ── 13. High entropy domain (random character generation) ─────────────
    const sld = parts.length >= 2 ? parts[parts.length - 2] : '';
    const entropy = this._shannonEntropy(sld);
    if (sld.length >= 8 && entropy > 3.5) {
      triggered.push({
        name: 'high_entropy_domain',
        label: 'Domain name appears randomly generated',
        severity: 'medium',
        weight: 1.5,
      });
    }

    // ── 14. Suspicious port ───────────────────────────────────────────────
    if (parsed.port && SUSPICIOUS_PORTS.has(parsed.port)) {
      triggered.push({
        name: 'suspicious_port',
        label: `Non-standard port ${parsed.port} — unusual for legitimate sites`,
        severity: 'medium',
        weight: 1.5,
      });
    }

    // ── 15. Too many dots in hostname ─────────────────────────────────────
    const dotCount = (hostname.match(/\./g) || []).length;
    if (dotCount >= 5) {
      triggered.push({
        name: 'excessive_dots',
        label: `${dotCount} dots in hostname — used to obscure the real domain`,
        severity: 'high',
        weight: 1.8,
      });
    }

    // ── 16. Encoded characters in URL (obfuscation) ───────────────────────
    const encodedCount = (url.match(/%[0-9a-fA-F]{2}/g) || []).length;
    if (encodedCount >= 5) {
      triggered.push({
        name: 'encoded_chars',
        label: `${encodedCount} encoded characters — possible URL obfuscation`,
        severity: 'medium',
        weight: 1.2,
      });
    }

    // ── 17. Deep path (excessive directories) ─────────────────────────────
    const pathDepth = path.split('/').filter(Boolean).length;
    if (pathDepth >= 6) {
      triggered.push({
        name: 'deep_path',
        label: `Deep URL path (${pathDepth} levels) — uncommon for legitimate sites`,
        severity: 'low',
        weight: 0.8,
      });
    }

    // ── 18. Sensitive keywords in query parameters ────────────────────────
    const sensitiveParams = ['password', 'passwd', 'token', 'session', 'ssn', 'credit', 'card'];
    const queryLower = search.toLowerCase();
    const matchedParams = sensitiveParams.filter(p => queryLower.includes(p));
    if (matchedParams.length > 0) {
      triggered.push({
        name: 'sensitive_query_params',
        label: `Sensitive data in URL query: ${matchedParams.join(', ')}`,
        severity: 'high',
        weight: 2.0,
      });
    }

    // ── 19. Data URI (embedded content) ───────────────────────────────────
    if (fullUrl.startsWith('data:')) {
      triggered.push({
        name: 'data_uri',
        label: 'Data URI — embedded content, not a real website',
        severity: 'critical',
        weight: 4.0,
      });
    }

    // ── 20. Punycode / IDN homograph attack ───────────────────────────────
    if (hostname.startsWith('xn--')) {
      triggered.push({
        name: 'punycode_idn',
        label: 'Internationalized domain (Punycode) — possible homograph attack',
        severity: 'high',
        weight: 2.0,
      });
    }

    // ── Compute final score ───────────────────────────────────────────────
    const rawScore = triggered.reduce((sum, f) => sum + f.weight, 0);
    const score = this._sigmoid(rawScore, 1.0, 3.5);
    const verdict = this._scoreToVerdict(score);

    // Sort features by weight descending for display
    triggered.sort((a, b) => b.weight - a.weight);

    return {
      score: Math.round(score * 1000) / 1000,
      verdict,
      features: triggered,
      rawScore: Math.round(rawScore * 100) / 100,
      url,
    };
  }


  // ═══════════════════════════════════════════════════════════════════════════
  //  Private helpers
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Sigmoid normalisation: maps raw score to [0, 1].
   * k controls steepness, x0 is the midpoint (score = 0.5).
   *
   * Calibration:
   *   rawScore 0   → ~0.03 (safe)
   *   rawScore 2   → ~0.18 (safe)
   *   rawScore 3.5 → ~0.50 (borderline)
   *   rawScore 5   → ~0.82 (phishing)
   *   rawScore 7   → ~0.97 (definite phishing)
   */
  _sigmoid(x, k, x0) {
    return 1 / (1 + Math.exp(-k * (x - x0)));
  }

  /** Map normalised score to a verdict string. */
  _scoreToVerdict(score) {
    if (score >= 0.65) return 'phishing';
    if (score >= 0.40) return 'suspicious';
    return 'safe';
  }

  /** Check if hostname is an IPv4 or IPv6 address. */
  _isIPAddress(hostname) {
    // IPv4 pattern
    if (/^(\d{1,3}\.){3}\d{1,3}$/.test(hostname)) return true;
    // IPv6 pattern (including brackets)
    if (/^\[?[0-9a-fA-F:]+\]?$/.test(hostname)) return true;
    // Hex/Octal encoded IP
    if (/^0x[0-9a-fA-F]+$/i.test(hostname)) return true;
    return false;
  }

  /**
   * Check if the hostname matches a typosquatting variant of a known brand.
   * Uses both exact variant matching and Levenshtein distance.
   */
  _checkTyposquatting(hostname) {
    const cleanHost = hostname.replace(/^www\./, '');
    const sld = cleanHost.split('.').slice(0, -1).join('.'); // everything before TLD

    for (const brand of BRAND_TARGETS) {
      // Skip if it's actually the legitimate domain
      if (cleanHost === brand.legit || cleanHost.endsWith('.' + brand.legit)) {
        return null;
      }

      // Check known typosquatting variants
      for (const variant of brand.variants) {
        if (sld.includes(variant)) {
          return { brand: brand.brand, legit: brand.legit, matched: variant };
        }
      }

      // Levenshtein check against the brand name
      const brandName = brand.legit.split('.')[0]; // e.g. "paypal"
      if (sld.length >= brandName.length - 1 && sld.length <= brandName.length + 4) {
        const distance = this._levenshtein(sld, brandName);
        if (distance > 0 && distance <= 2) {
          return { brand: brand.brand, legit: brand.legit, matched: sld };
        }
      }
    }
    return null;
  }

  /** Levenshtein edit distance between two strings. */
  _levenshtein(a, b) {
    const m = a.length;
    const n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,      // deletion
          dp[i][j - 1] + 1,      // insertion
          dp[i - 1][j - 1] + cost // substitution
        );
      }
    }
    return dp[m][n];
  }

  /**
   * Shannon entropy of a string — high values indicate randomness.
   * Legitimate domains tend to be human-readable (low entropy ~2.5).
   * Randomly generated domains have entropy > 3.5.
   */
  _shannonEntropy(str) {
    if (!str || str.length === 0) return 0;
    const freq = {};
    for (const ch of str) {
      freq[ch] = (freq[ch] || 0) + 1;
    }
    let entropy = 0;
    const len = str.length;
    for (const count of Object.values(freq)) {
      const p = count / len;
      if (p > 0) entropy -= p * Math.log2(p);
    }
    return entropy;
  }
}
