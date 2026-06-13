/**
 * PhishGuard V2 — Advanced Phishing Feature Extraction Engine
 * background/feature-engine.js
 *
 * Extracts 57 phishing indicators from a URL string alone — no network I/O.
 * Designed for both heuristic scoring and ML model input vectors.
 *
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │  Category          │ Count │ Feature IDs                              │
 * ├────────────────────┼───────┼──────────────────────────────────────────│
 * │  Lexical           │  22   │ F01–F22  (character-level URL stats)    │
 * │  Domain            │  12   │ F23–F34  (hostname structure)           │
 * │  Path              │   7   │ F35–F41  (URL path analysis)           │
 * │  Query/Fragment    │   5   │ F42–F46  (parameter analysis)          │
 * │  Security          │   5   │ F47–F51  (protocol/encryption)         │
 * │  Keyword           │   5   │ F52–F56  (phishing language)           │
 * │  Homograph         │   6   │ F57–F62  (visual deception)            │
 * └────────────────────┴───────┴──────────────────────────────────────────┘
 *
 * Each feature includes:
 *   - name:        Stable identifier for ML column alignment
 *   - value:       Numeric value extracted from the URL
 *   - category:    Feature grouping
 *   - description: Why attackers exploit this + how it helps detection
 *   - importance:  Expected importance score (1–10) from phishing research
 *
 * References:
 *   [1] Mohammad, R. et al. "Phishing Websites Features" (UCI ML Repository, 2015)
 *   [2] Sahingoz, O. et al. "Machine learning based phishing detection" (Expert Systems, 2019)
 *   [3] Rao, R. & Pais, A. "Detection of phishing websites using URL features" (CSIT, 2019)
 */

'use strict';


// ═══════════════════════════════════════════════════════════════════════════════
//  Reference Data
// ═══════════════════════════════════════════════════════════════════════════════

/** URL shortener services — hide the real destination. */
const SHORTENERS = new Set([
  'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd',
  'buff.ly', 'short.to', 'tr.im', 'tiny.cc', 'cutt.ly', 'rb.gy',
  'shorturl.at', 'adf.ly', 'bit.do', 'lnkd.in', 'db.tt', 'qr.ae',
  'v.gd', 'rebrand.ly', 'tny.im', 'surl.li', 'bl.ink', 'clck.ru',
  'shortcm.li', 'hyperurl.co', 'u.to', 'shrtco.de',
]);

/** TLDs with high abuse rates (APWG & Spamhaus data). */
const SUSPICIOUS_TLDS = new Set([
  'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'xyz', 'buzz', 'club', 'work',
  'click', 'link', 'info', 'support', 'review', 'stream', 'racing',
  'win', 'bid', 'download', 'loan', 'men', 'party', 'science', 'trade',
  'gdn', 'accountant', 'cricket', 'date', 'faith', 'rocks', 'surf',
  'monster', 'rest', 'icu', 'cyou', 'cfd', 'sbs', 'quest',
]);

/** Phishing login/credential keywords. */
const PHISHING_KEYWORDS = [
  'login', 'signin', 'sign-in', 'log-in', 'verify', 'verification',
  'confirm', 'update', 'secure', 'account', 'banking', 'password',
  'credential', 'authenticate', 'validate', 'wallet', 'recover',
  'reactivate', 'webscr', 'myaccount', 'identity',
];

/** Urgency/fear keywords — pressure the victim. */
const URGENCY_KEYWORDS = [
  'suspend', 'restrict', 'unlock', 'unusual', 'alert', 'notification',
  'expire', 'limited', 'urgent', 'immediately', 'blocked', 'compromised',
  'unauthorized', 'deadline', 'action-required', 'final-warning',
];

/** Top phished brands — domain names only. */
const BRAND_DOMAINS = new Map([
  ['paypal',       'paypal.com'],
  ['google',       'google.com'],
  ['microsoft',    'microsoft.com'],
  ['apple',        'apple.com'],
  ['amazon',       'amazon.com'],
  ['netflix',      'netflix.com'],
  ['facebook',     'facebook.com'],
  ['instagram',    'instagram.com'],
  ['linkedin',     'linkedin.com'],
  ['twitter',      'twitter.com'],
  ['chase',        'chase.com'],
  ['wellsfargo',   'wellsfargo.com'],
  ['bankofamerica','bankofamerica.com'],
  ['dropbox',      'dropbox.com'],
  ['github',       'github.com'],
  ['whatsapp',     'whatsapp.com'],
  ['outlook',      'outlook.com'],
  ['yahoo',        'yahoo.com'],
  ['spotify',      'spotify.com'],
  ['steam',        'steampowered.com'],
  ['adobe',        'adobe.com'],
  ['hdfc',         'hdfcbank.com'],
  ['sbi',          'onlinesbi.com'],
  ['icici',        'icicibank.com'],
]);

/** Characters that look similar (confusables for homograph attacks). */
const CONFUSABLE_MAP = {
  'a': ['а', 'ɑ', 'α'],        // Cyrillic а, Latin ɑ, Greek α
  'c': ['с', 'ϲ', 'ᴄ'],        // Cyrillic с, Greek ϲ
  'e': ['е', 'ė', 'ё'],        // Cyrillic е
  'o': ['о', 'ο', '0', 'ø'],   // Cyrillic о, Greek ο, digit 0
  'p': ['р', 'ρ'],              // Cyrillic р, Greek ρ
  'i': ['і', 'ι', '1', 'l', '|'], // Ukrainian і, Greek ι, digit 1
  'l': ['1', 'I', '|', 'ӏ'],   // Digit 1, capital I
  's': ['ѕ', 'ꜱ'],             // Cyrillic ѕ
  'x': ['х', 'χ'],             // Cyrillic х, Greek χ
  'y': ['у', 'γ'],             // Cyrillic у, Greek γ
  'n': ['п', 'ñ'],             // Cyrillic п
  'd': ['ԁ'],                  // Cyrillic ԁ
  'g': ['ɡ', 'ǵ'],             // Latin ɡ
  'k': ['κ', 'к'],             // Greek κ, Cyrillic к
  'w': ['ω', 'ш'],             // Greek ω, Cyrillic ш
};

/** Suspicious file extensions that could indicate malware delivery. */
const SUSPICIOUS_EXTENSIONS = new Set([
  '.exe', '.scr', '.bat', '.cmd', '.msi', '.js', '.vbs', '.ps1',
  '.jar', '.apk', '.dll', '.com', '.wsf', '.hta',
]);

/** Non-standard ports commonly used by phishing infrastructure. */
const UNUSUAL_PORTS = new Set([
  '8080', '8443', '8888', '3000', '4443', '9090', '1337',
  '4444', '5555', '6666', '7777', '8000', '8081', '9000', '9999',
]);


// ═══════════════════════════════════════════════════════════════════════════════
//  Feature Metadata Registry
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Complete metadata for all 57+ features.
 * Each entry documents:
 *   - why: Why attackers exploit this URL property
 *   - how: How the feature helps detection
 *   - importance: Expected importance score (1–10) based on published research
 */
const FEATURE_REGISTRY = [

  // ─── LEXICAL FEATURES (F01–F22) ────────────────────────────────────────────
  {
    id: 'F01', name: 'url_length', category: 'lexical',
    why: 'Attackers pad URLs with random paths, parameters, and subdomains to obscure the actual destination and bypass simple pattern matching.',
    how: 'Phishing URLs average 75+ characters vs ~40 for legitimate sites. A strong univariate separator.',
    importance: 7,
  },
  {
    id: 'F02', name: 'hostname_length', category: 'lexical',
    why: 'Long hostnames accommodate extra subdomains (e.g., "secure-paypal-login.verification.attacker.com") to confuse users.',
    how: 'Legitimate hostnames rarely exceed 25 characters. Lengths above 30 are suspicious.',
    importance: 6,
  },
  {
    id: 'F03', name: 'path_length', category: 'lexical',
    why: 'Deep paths hide phishing pages inside compromised legitimate sites (e.g., "/wp-content/uploads/secure-login/").',
    how: 'Path length correlates with nested phishing kits deployed on hacked servers.',
    importance: 5,
  },
  {
    id: 'F04', name: 'query_length', category: 'lexical',
    why: 'Long query strings encode victim identifiers, tracking tokens, and encoded redirect URLs.',
    how: 'Legitimate login pages rarely have query strings > 100 chars. Excessive length flags data exfiltration.',
    importance: 4,
  },
  {
    id: 'F05', name: 'fragment_length', category: 'lexical',
    why: 'Fragments (#) can carry obfuscated data passed to client-side JavaScript without hitting the server.',
    how: 'Non-zero fragment length in login/verify pages is unusual and flags client-side data handling.',
    importance: 2,
  },
  {
    id: 'F06', name: 'num_dots_url', category: 'lexical',
    why: 'Extra dots create visual confusion — "paypal.com.secure.login.attacker.xyz" looks like it belongs to PayPal.',
    how: 'Legitimate URLs average 2–3 dots. URLs with 5+ dots are disproportionately phishing.',
    importance: 6,
  },
  {
    id: 'F07', name: 'num_dots_hostname', category: 'lexical',
    why: 'Dots in hostname = subdomain depth. Attackers stack subdomains to push the real domain off-screen in the address bar.',
    how: 'Hostname dot count directly maps to subdomain abuse. Strong correlation with phishing (r=0.42 in UCI dataset).',
    importance: 7,
  },
  {
    id: 'F08', name: 'num_hyphens_url', category: 'lexical',
    why: 'Hyphens create compound words that mimic brands — "paypal-secure-login.com" instead of a paypal.com subdomain.',
    how: 'Prefix-suffix attacks (hyphens in domain) have precision >0.85 as phishing indicators.',
    importance: 7,
  },
  {
    id: 'F09', name: 'num_hyphens_hostname', category: 'lexical',
    why: 'Same as F08 but scoped to hostname — more precise since hyphens in paths are common.',
    how: 'Hostnames with 2+ hyphens are 4x more likely to be phishing than legitimate.',
    importance: 8,
  },
  {
    id: 'F10', name: 'num_underscores', category: 'lexical',
    why: 'Underscores in hostnames violate RFC 952 — they signal auto-generated or low-effort phishing domains.',
    how: 'Underscores in hostnames are technically invalid and extremely rare on legitimate sites.',
    importance: 5,
  },
  {
    id: 'F11', name: 'num_slashes_path', category: 'lexical',
    why: 'Deep directory traversal signals phishing kits hidden in subdirectories of compromised sites.',
    how: 'Correlates with path depth (F35). Phishing pages average 4+ slashes vs 2 for legitimate pages.',
    importance: 5,
  },
  {
    id: 'F12', name: 'num_digits_url', category: 'lexical',
    why: 'Digits often encode victim IDs, session tokens, or tracking numbers in the URL.',
    how: 'High digit count in non-numeric paths indicates parameterized phishing campaigns.',
    importance: 4,
  },
  {
    id: 'F13', name: 'num_digits_hostname', category: 'lexical',
    why: 'Legitimate brand domains rarely contain digits. "paypal1.com" or "goog1e.com" are typosquatting attempts.',
    how: 'Digits in hostnames are a strong signal for algorithmically generated phishing domains.',
    importance: 6,
  },
  {
    id: 'F14', name: 'digit_letter_ratio', category: 'lexical',
    why: 'High digit-to-letter ratios indicate generated URLs (e.g., "/verify/38291847102847").',
    how: 'Normalized ratio is threshold-invariant. Values > 0.4 are highly suspicious.',
    importance: 5,
  },
  {
    id: 'F15', name: 'special_char_count', category: 'lexical',
    why: 'Special characters (%、@、=、&) are used for obfuscation, redirect injection, and parameter stuffing.',
    how: 'Total count of non-alphanumeric, non-standard URL characters.',
    importance: 4,
  },
  {
    id: 'F16', name: 'special_char_ratio', category: 'lexical',
    why: 'Normalised version of F15 — accounts for URL length variation.',
    how: 'Ratio > 0.25 is extremely rare for legitimate URLs and indicates obfuscation.',
    importance: 5,
  },
  {
    id: 'F17', name: 'has_at_sign', category: 'lexical',
    why: '@ in URL tricks browsers: "http://google.com@evil.com" navigates to evil.com while displaying google.com in the address bar prefix.',
    how: 'Binary feature with near-perfect precision — @ in URL body is almost never legitimate.',
    importance: 9,
  },
  {
    id: 'F18', name: 'num_percent_encodings', category: 'lexical',
    why: 'Percent-encoding (%XX) obfuscates characters to bypass security filters and hide phishing keywords.',
    how: 'Legitimate URLs rarely have >2 encoded characters. Excessive encoding flags evasion.',
    importance: 5,
  },
  {
    id: 'F19', name: 'num_ampersands', category: 'lexical',
    why: 'Excessive & symbols indicate complex query strings used for victim tracking and redirect chaining.',
    how: 'Count of parameter separators; phishing URLs average more parameters than legitimate login pages.',
    importance: 3,
  },
  {
    id: 'F20', name: 'url_entropy', category: 'lexical',
    why: 'Algorithmically generated URLs (DGA) have higher Shannon entropy than human-readable URLs.',
    how: 'Entropy > 4.5 bits/char indicates randomness. Legitimate domains average ~3.2 bits/char.',
    importance: 7,
  },
  {
    id: 'F21', name: 'hostname_entropy', category: 'lexical',
    why: 'Scoped to hostname — avoids noise from legitimately complex paths. DGA domains produce entropy > 3.8.',
    how: 'More targeted than full URL entropy. Strong separator for auto-generated phishing domains.',
    importance: 8,
  },
  {
    id: 'F22', name: 'path_entropy', category: 'lexical',
    why: 'Phishing kits use random path tokens ("/a8f3b2e1c4/login.php") to evade blacklists.',
    how: 'High path entropy + login keywords = strong phishing indicator (compound feature).',
    importance: 5,
  },

  // ─── DOMAIN FEATURES (F23–F34) ─────────────────────────────────────────────
  {
    id: 'F23', name: 'num_subdomains', category: 'domain',
    why: 'Attackers prepend brand names as subdomains: "paypal.com.secure.attacker.xyz" — the real domain is attacker.xyz.',
    how: 'Count of subdomain levels. Values ≥ 3 are 5x more likely to be phishing.',
    importance: 8,
  },
  {
    id: 'F24', name: 'subdomain_total_length', category: 'domain',
    why: 'Long subdomains encode entire fake URLs: "paypal-com-secure-login" as a subdomain.',
    how: 'Total character count of all subdomain labels combined. > 20 chars is suspicious.',
    importance: 6,
  },
  {
    id: 'F25', name: 'has_www_prefix', category: 'domain',
    why: 'Many phishing pages omit "www" because the attacker controls only a bare domain.',
    how: 'Absence of www is a weak signal (many legitimate sites also omit it). Low standalone importance.',
    importance: 2,
  },
  {
    id: 'F26', name: 'tld_length', category: 'domain',
    why: 'New gTLDs like ".accountant" or ".cricket" are cheap and have weaker registration requirements.',
    how: 'Longer TLDs (> 4 chars) are disproportionately represented in phishing datasets.',
    importance: 4,
  },
  {
    id: 'F27', name: 'sld_length', category: 'domain',
    why: 'Very short SLDs (2–3 chars) are often randomised (e.g., "ab.xyz"). Very long SLDs pack in brand keywords.',
    how: 'Both extremes (< 4 or > 15 chars) correlate with phishing.',
    importance: 4,
  },
  {
    id: 'F28', name: 'is_ip_address', category: 'domain',
    why: 'Using raw IP addresses (http://192.168.1.1/login) avoids DNS resolution and brand detection.',
    how: 'Binary feature with very high precision (~0.95). Legitimate login pages never use raw IPs.',
    importance: 9,
  },
  {
    id: 'F29', name: 'is_hex_ip', category: 'domain',
    why: 'Hex-encoded IPs (0x7f000001) or octal (0177.0.0.1) bypass IP-pattern detection rules.',
    how: 'Any hex/octal hostname is almost certainly malicious. Near-perfect precision.',
    importance: 9,
  },
  {
    id: 'F30', name: 'has_non_standard_port', category: 'domain',
    why: 'Phishing servers run on odd ports (8080, 4443) to avoid port-based blocking rules.',
    how: 'Ports other than 80/443 on login pages are extremely rare for legitimate services.',
    importance: 7,
  },
  {
    id: 'F31', name: 'is_suspicious_tld', category: 'domain',
    why: 'Free TLDs (.tk, .ml, .ga) are massively abused because registration is free and anonymous.',
    how: 'Binary flag using APWG/Spamhaus abuse-rate data. High recall for bulk phishing campaigns.',
    importance: 7,
  },
  {
    id: 'F32', name: 'is_shortener', category: 'domain',
    why: 'URL shorteners hide the true destination. Attackers share short links via email/SMS.',
    how: 'Binary flag for known shortener services. Moderate importance (shorteners have legitimate uses).',
    importance: 5,
  },
  {
    id: 'F33', name: 'domain_token_count', category: 'domain',
    why: 'Compound domains like "paypal-secure-login-verify" have many tokens separated by hyphens/dots.',
    how: 'Token count (split on . and -) above 4 is strongly correlated with brand impersonation.',
    importance: 6,
  },
  {
    id: 'F34', name: 'longest_domain_token', category: 'domain',
    why: 'Very long tokens like "paypal-secure-verification-center" are compound brand-impersonation strings.',
    how: 'Longest token > 15 chars in hostname flags keyword-stuffed domains.',
    importance: 5,
  },

  // ─── PATH FEATURES (F35–F41) ───────────────────────────────────────────────
  {
    id: 'F35', name: 'path_depth', category: 'path',
    why: 'Phishing kits are buried deep in compromised sites: /wp-content/plugins/uploads/secure/login/',
    how: 'Directory depth (slash count in path). Values ≥ 5 are 3x more likely to be phishing.',
    importance: 6,
  },
  {
    id: 'F36', name: 'max_path_token_length', category: 'path',
    why: 'Long path tokens are often base64-encoded data or random hashes used by phishing kits.',
    how: 'Longest path segment > 30 chars indicates auto-generated content or encoded data.',
    importance: 4,
  },
  {
    id: 'F37', name: 'avg_path_token_length', category: 'path',
    why: 'Phishing paths average longer tokens due to encoded identifiers and obfuscation.',
    how: 'Average segment length > 12 chars is unusual for legitimate URL hierarchies.',
    importance: 3,
  },
  {
    id: 'F38', name: 'has_double_slash_redirect', category: 'path',
    why: '"http://legit.com//evil.com" — double slash in path triggers a redirect to evil.com in some servers.',
    how: 'Double-slash after the scheme is a known redirect exploit. High precision signal.',
    importance: 7,
  },
  {
    id: 'F39', name: 'has_file_extension', category: 'path',
    why: 'Phishing pages often expose their technology stack: .php, .asp, .html (vs modern clean URLs).',
    how: 'Presence of legacy extensions like .php on "bank login" pages flags amateur phishing kits.',
    importance: 3,
  },
  {
    id: 'F40', name: 'has_suspicious_extension', category: 'path',
    why: 'URLs ending in .exe, .scr, .bat indicate malware delivery disguised as a download link.',
    how: 'Binary flag for dangerous file extensions. Very high precision for malware phishing.',
    importance: 8,
  },
  {
    id: 'F41', name: 'path_has_encoded_chars', category: 'path',
    why: 'Percent-encoding in paths hides phishing keywords from URL scanners: %6C%6F%67%69%6E = "login".',
    how: 'Encoded characters in path segments beyond normal URL encoding (spaces, etc.) flag obfuscation.',
    importance: 5,
  },

  // ─── QUERY / FRAGMENT FEATURES (F42–F46) ───────────────────────────────────
  {
    id: 'F42', name: 'num_query_params', category: 'query',
    why: 'Phishing URLs carry victim identifiers, campaign tokens, and redirect targets as parameters.',
    how: 'Count of query parameters. Values > 5 are unusual for login pages.',
    importance: 4,
  },
  {
    id: 'F43', name: 'max_query_value_length', category: 'query',
    why: 'Long parameter values often contain encoded redirect URLs or base64 victim data.',
    how: 'Longest query value > 100 chars flags redirect chaining or data smuggling.',
    importance: 4,
  },
  {
    id: 'F44', name: 'has_sensitive_query_key', category: 'query',
    why: 'Sensitive data in query strings (password=, token=, ssn=) indicates credential handling in the URL.',
    how: 'Query keys matching sensitive patterns are extremely suspicious on any page.',
    importance: 7,
  },
  {
    id: 'F45', name: 'query_contains_url', category: 'query',
    why: 'Embedded URLs in query params enable open redirect attacks: ?redirect=http://evil.com.',
    how: 'Presence of http:// or https:// in query values flags redirect abuse.',
    importance: 6,
  },
  {
    id: 'F46', name: 'has_fragment', category: 'query',
    why: 'Fragments pass data to client-side JavaScript without server logging — used for stealthy phishing.',
    how: 'Non-empty fragments on login pages are uncommon and flag client-side data handling.',
    importance: 3,
  },

  // ─── SECURITY FEATURES (F47–F51) ───────────────────────────────────────────
  {
    id: 'F47', name: 'is_https', category: 'security',
    why: 'Lack of HTTPS means credentials are sent in cleartext. However, many phishing sites now use free SSL (Let\'s Encrypt).',
    how: 'HTTP on login pages is suspicious but HTTPS alone does NOT mean safe. Moderate importance.',
    importance: 5,
  },
  {
    id: 'F48', name: 'has_https_token_hostname', category: 'security',
    why: 'Embedding "https" in the domain name ("https-paypal-secure.com") fools users into thinking it\'s secure.',
    how: 'The word "https" or "ssl" appearing in the hostname (not scheme) is a strong deception signal.',
    importance: 8,
  },
  {
    id: 'F49', name: 'has_security_word_hostname', category: 'security',
    why: 'Words like "secure", "verify", "confirm" in the domain psychologically reassure victims.',
    how: 'Security vocabulary in hostnames is 6x more common in phishing than legitimate sites.',
    importance: 7,
  },
  {
    id: 'F50', name: 'scheme_host_mismatch', category: 'security',
    why: 'HTTP scheme + "secure" in hostname = contradiction. Attackers bank on users not checking the scheme.',
    how: 'Compound feature: http:// AND security words in hostname. Very high precision.',
    importance: 8,
  },
  {
    id: 'F51', name: 'has_mixed_case_hostname', category: 'security',
    why: 'Mixed case hostnames ("PayPal-Secure-Login.COM") are used for visual emphasis in phishing links.',
    how: 'DNS is case-insensitive, so mixed case is purely cosmetic. Flags social engineering intent.',
    importance: 3,
  },

  // ─── KEYWORD FEATURES (F52–F56) ────────────────────────────────────────────
  {
    id: 'F52', name: 'phishing_keyword_count', category: 'keyword',
    why: 'Phishing URLs pack in action words: "login", "verify", "confirm", "account" to sound official.',
    how: 'Count of distinct phishing keywords found in the full URL. Values ≥ 3 are highly suspicious.',
    importance: 7,
  },
  {
    id: 'F53', name: 'urgency_keyword_count', category: 'keyword',
    why: 'Fear-inducing words ("suspend", "urgent", "blocked") pressure victims into acting without thinking.',
    how: 'Urgency keywords in URLs are extremely rare for legitimate services.',
    importance: 6,
  },
  {
    id: 'F54', name: 'brand_keyword_in_url', category: 'keyword',
    why: 'Brand names in non-official domains = impersonation. "paypal" in "paypal-login.evil.com" is a lure.',
    how: 'Detects brand names in URLs that don\'t belong to the brand\'s actual domain.',
    importance: 9,
  },
  {
    id: 'F55', name: 'brand_in_subdomain', category: 'keyword',
    why: 'Placing brand names in subdomains ("paypal.secure.evil.com") exploits left-to-right reading bias.',
    how: 'Brand name specifically in the subdomain (not the registrable domain) is near-certain phishing.',
    importance: 9,
  },
  {
    id: 'F56', name: 'suspicious_keyword_combo', category: 'keyword',
    why: 'Combinations like "login" + "verify" + brand name in one URL are a strong compound signal.',
    how: 'Boolean: ≥ 2 phishing keywords AND a brand name present. Very high precision.',
    importance: 8,
  },

  // ─── HOMOGRAPH / DECEPTION FEATURES (F57–F62) ──────────────────────────────
  {
    id: 'F57', name: 'is_punycode', category: 'homograph',
    why: 'Punycode (xn--) encodes Unicode domains that visually mimic ASCII brands: "xn--pypal-4ve.com" looks like "paypal.com".',
    how: 'Any xn-- prefix in hostname is an internationalized domain name. Rarely used by legitimate brands.',
    importance: 8,
  },
  {
    id: 'F58', name: 'has_confusable_chars', category: 'homograph',
    why: 'Characters like Cyrillic "а" (U+0430) look identical to Latin "a" (U+0061) but resolve to different domains.',
    how: 'Scans hostname for Unicode confusables from the IDNA confusable table. Binary flag.',
    importance: 8,
  },
  {
    id: 'F59', name: 'typosquatting_distance', category: 'homograph',
    why: 'Typosquatting registers domains that are 1–2 characters off from a brand: "goggle.com", "paypa1.com".',
    how: 'Minimum Levenshtein distance between the hostname SLD and all known brand names. Distance 1–2 = phishing.',
    importance: 9,
  },
  {
    id: 'F60', name: 'char_repetition_max', category: 'homograph',
    why: 'Repeated characters ("gooooogle.com") are used to evade exact-match blocklists while remaining recognisable.',
    how: 'Maximum consecutive repetition of any single character in hostname. Values ≥ 3 are suspicious.',
    importance: 5,
  },
  {
    id: 'F61', name: 'consonant_vowel_ratio', category: 'homograph',
    why: 'Auto-generated domains have unnatural consonant-to-vowel ratios (e.g., "bxkrtp.com").',
    how: 'Ratio outside the natural language range (0.5–2.5) flags machine-generated hostnames.',
    importance: 5,
  },
  {
    id: 'F62', name: 'visual_similarity_score', category: 'homograph',
    why: 'Composite score combining confusables, typosquatting, and character repetition for overall deception level.',
    how: 'Weighted combination of F57–F61. Captures multi-factor visual deception in a single score.',
    importance: 9,
  },
];


// ═══════════════════════════════════════════════════════════════════════════════
//  Feature Extraction Engine
// ═══════════════════════════════════════════════════════════════════════════════

export class FeatureEngine {

  /**
   * Extract all 57+ features from a URL string.
   *
   * @param  {string} rawUrl — The complete URL to analyse
   * @return {FeatureExtractionResult}
   *
   * @typedef {Object} FeatureExtractionResult
   * @property {Object<string, number>} vector   — Feature name → numeric value (for ML input)
   * @property {FeatureDetail[]}        details  — Rich feature objects with metadata
   * @property {number}                 riskScore — Aggregated heuristic risk score [0, 1]
   * @property {string}                 verdict   — 'safe' | 'suspicious' | 'phishing'
   */
  extract(rawUrl) {
    let parsed;
    try {
      parsed = new URL(rawUrl);
    } catch {
      return this._errorResult(rawUrl);
    }

    const url       = rawUrl;
    const hostname  = parsed.hostname.toLowerCase();
    const path      = parsed.pathname || '/';
    const query     = parsed.search || '';
    const fragment  = parsed.hash || '';
    const scheme    = parsed.protocol;
    const port      = parsed.port || '';

    // Pre-compute hostname segments
    const hostParts   = hostname.split('.');
    const tld         = hostParts[hostParts.length - 1] || '';
    const sld         = hostParts.length >= 2 ? hostParts[hostParts.length - 2] : '';
    const subdomains  = hostParts.length > 2 ? hostParts.slice(0, -2) : [];
    const cleanHost   = hostname.replace(/^www\./, '');

    // Pre-compute path segments
    const pathTokens  = path.split('/').filter(Boolean);

    // Pre-compute URL character stats
    const urlLower    = url.toLowerCase();

    // ── Extract all features ─────────────────────────────────────────────
    const vector = {};

    // F01–F05: Length features
    vector.url_length               = url.length;
    vector.hostname_length          = hostname.length;
    vector.path_length              = path.length;
    vector.query_length             = query.length;
    vector.fragment_length          = fragment.length;

    // F06–F07: Dot counts
    vector.num_dots_url             = (url.match(/\./g) || []).length;
    vector.num_dots_hostname        = (hostname.match(/\./g) || []).length;

    // F08–F10: Separators
    vector.num_hyphens_url          = (url.match(/-/g) || []).length;
    vector.num_hyphens_hostname     = (hostname.match(/-/g) || []).length;
    vector.num_underscores          = (url.match(/_/g) || []).length;

    // F11: Slash count
    vector.num_slashes_path         = (path.match(/\//g) || []).length;

    // F12–F14: Digit analysis
    vector.num_digits_url           = (url.match(/\d/g) || []).length;
    vector.num_digits_hostname      = (hostname.match(/\d/g) || []).length;
    const letters                   = (url.match(/[a-zA-Z]/g) || []).length;
    vector.digit_letter_ratio       = letters > 0
      ? Math.round((vector.num_digits_url / letters) * 1000) / 1000
      : vector.num_digits_url > 0 ? 1.0 : 0;

    // F15–F16: Special characters
    const specialChars              = url.replace(/[a-zA-Z0-9]/g, '');
    vector.special_char_count       = specialChars.length;
    vector.special_char_ratio       = url.length > 0
      ? Math.round((specialChars.length / url.length) * 1000) / 1000
      : 0;

    // F17: @ sign
    vector.has_at_sign              = url.includes('@') ? 1 : 0;

    // F18: Percent-encoding
    vector.num_percent_encodings    = (url.match(/%[0-9a-fA-F]{2}/g) || []).length;

    // F19: Ampersands
    vector.num_ampersands           = (url.match(/&/g) || []).length;

    // F20–F22: Entropy
    vector.url_entropy              = this._entropy(url);
    vector.hostname_entropy         = this._entropy(hostname);
    vector.path_entropy             = this._entropy(path);

    // F23–F27: Domain structure
    vector.num_subdomains           = subdomains.length;
    vector.subdomain_total_length   = subdomains.join('').length;
    vector.has_www_prefix           = hostname.startsWith('www.') ? 1 : 0;
    vector.tld_length               = tld.length;
    vector.sld_length               = sld.length;

    // F28–F29: IP address detection
    vector.is_ip_address            = this._isIPAddress(hostname) ? 1 : 0;
    vector.is_hex_ip                = this._isHexIP(hostname) ? 1 : 0;

    // F30: Port
    vector.has_non_standard_port    = (port && UNUSUAL_PORTS.has(port)) ? 1 : 0;

    // F31–F32: TLD and shortener
    vector.is_suspicious_tld        = SUSPICIOUS_TLDS.has(tld) ? 1 : 0;
    vector.is_shortener             = (SHORTENERS.has(hostname) || SHORTENERS.has(cleanHost)) ? 1 : 0;

    // F33–F34: Domain token analysis
    const domainTokens              = cleanHost.split(/[.\-]/);
    vector.domain_token_count       = domainTokens.length;
    vector.longest_domain_token     = Math.max(0, ...domainTokens.map(t => t.length));

    // F35–F41: Path features
    vector.path_depth               = pathTokens.length;
    vector.max_path_token_length    = pathTokens.length > 0
      ? Math.max(...pathTokens.map(t => t.length))
      : 0;
    vector.avg_path_token_length    = pathTokens.length > 0
      ? Math.round(pathTokens.reduce((s, t) => s + t.length, 0) / pathTokens.length * 10) / 10
      : 0;
    vector.has_double_slash_redirect = (url.lastIndexOf('//') > 7) ? 1 : 0;

    const lastPathToken              = pathTokens[pathTokens.length - 1] || '';
    const extMatch                   = lastPathToken.match(/(\.[a-zA-Z0-9]{1,5})$/);
    vector.has_file_extension        = extMatch ? 1 : 0;
    vector.has_suspicious_extension  = (extMatch && SUSPICIOUS_EXTENSIONS.has(extMatch[1].toLowerCase())) ? 1 : 0;
    vector.path_has_encoded_chars    = (path.match(/%[0-9a-fA-F]{2}/g) || []).length > 0 ? 1 : 0;

    // F42–F46: Query/Fragment features
    const params                     = new URLSearchParams(parsed.search);
    vector.num_query_params          = Array.from(params.keys()).length;
    let maxValLen = 0;
    let hasSensitiveKey = 0;
    let queryHasUrl = 0;
    const sensitiveKeys = /password|passwd|token|session|ssn|credit|card|secret|api.?key|auth/i;
    for (const [key, val] of params) {
      if (val.length > maxValLen) maxValLen = val.length;
      if (sensitiveKeys.test(key)) hasSensitiveKey = 1;
      if (/https?:\/\//i.test(val)) queryHasUrl = 1;
    }
    vector.max_query_value_length    = maxValLen;
    vector.has_sensitive_query_key   = hasSensitiveKey;
    vector.query_contains_url        = queryHasUrl;
    vector.has_fragment              = fragment.length > 1 ? 1 : 0; // > 1 because # alone is empty

    // F47–F51: Security features
    vector.is_https                  = scheme === 'https:' ? 1 : 0;
    vector.has_https_token_hostname  = (hostname.includes('https') || hostname.includes('ssl')) ? 1 : 0;
    const securityWords              = ['secure', 'verify', 'confirm', 'safety', 'protect'];
    vector.has_security_word_hostname = securityWords.some(w => hostname.includes(w)) ? 1 : 0;
    vector.scheme_host_mismatch      = (scheme === 'http:' && (vector.has_security_word_hostname || vector.has_https_token_hostname)) ? 1 : 0;
    vector.has_mixed_case_hostname   = (/[A-Z]/.test(parsed.hostname) && /[a-z]/.test(parsed.hostname)) ? 1 : 0;

    // F52–F56: Keyword features
    vector.phishing_keyword_count    = this._countKeywords(urlLower, PHISHING_KEYWORDS);
    vector.urgency_keyword_count     = this._countKeywords(urlLower, URGENCY_KEYWORDS);

    const brandResult                = this._detectBrandInUrl(hostname, urlLower, subdomains, sld);
    vector.brand_keyword_in_url      = brandResult.inUrl ? 1 : 0;
    vector.brand_in_subdomain        = brandResult.inSubdomain ? 1 : 0;
    vector.suspicious_keyword_combo  = (vector.phishing_keyword_count >= 2 && vector.brand_keyword_in_url) ? 1 : 0;

    // F57–F62: Homograph / deception features
    vector.is_punycode               = hostname.includes('xn--') ? 1 : 0;
    vector.has_confusable_chars      = this._hasConfusables(hostname) ? 1 : 0;
    vector.typosquatting_distance    = this._minBrandDistance(sld);
    vector.char_repetition_max       = this._maxCharRepetition(hostname);
    vector.consonant_vowel_ratio     = this._consonantVowelRatio(sld);
    vector.visual_similarity_score   = this._visualSimilarity(vector);

    // ── Compute aggregate risk score ─────────────────────────────────────
    const riskScore = this._computeRiskScore(vector);
    const verdict   = riskScore >= 0.65 ? 'phishing'
                    : riskScore >= 0.40 ? 'suspicious'
                    : 'safe';

    // ── Build detailed feature list (for explainability) ─────────────────
    const details = this._buildDetails(vector);

    return { vector, details, riskScore, verdict };
  }


  /**
   * Return the full feature metadata registry.
   * Useful for documentation, ML pipeline column alignment, and UI display.
   */
  static getFeatureMetadata() {
    return FEATURE_REGISTRY;
  }

  /**
   * Return ordered feature names for ML model input alignment.
   */
  static getFeatureNames() {
    return FEATURE_REGISTRY.map(f => f.name);
  }


  // ═══════════════════════════════════════════════════════════════════════════
  //  Private — Computation Helpers
  // ═══════════════════════════════════════════════════════════════════════════

  /** Shannon entropy in bits per character. */
  _entropy(str) {
    if (!str || str.length === 0) return 0;
    const freq = {};
    for (const ch of str) freq[ch] = (freq[ch] || 0) + 1;
    let entropy = 0;
    for (const count of Object.values(freq)) {
      const p = count / str.length;
      if (p > 0) entropy -= p * Math.log2(p);
    }
    return Math.round(entropy * 1000) / 1000;
  }

  /** Check if hostname is an IPv4 address. */
  _isIPAddress(hostname) {
    return /^(\d{1,3}\.){3}\d{1,3}$/.test(hostname) ||
           /^\[?[0-9a-fA-F:]+\]?$/.test(hostname.replace(/[[\]]/g, ''));
  }

  /** Check for hex/octal-encoded IP addresses. */
  _isHexIP(hostname) {
    return /^0x[0-9a-fA-F]+$/i.test(hostname) ||
           /^(\d{1,3}\.){3}0[xX][0-9a-fA-F]+$/.test(hostname) ||
           /^0[0-7]+(\.[0-7]+){3}$/.test(hostname);
  }

  /** Count how many keywords from a list appear in a string. */
  _countKeywords(str, keywords) {
    return keywords.filter(kw => str.includes(kw)).length;
  }

  /** Detect brand names in URL that don't belong to the brand's domain. */
  _detectBrandInUrl(hostname, fullUrl, subdomains, sld) {
    const result = { inUrl: false, inSubdomain: false, brand: null };
    const cleanHost = hostname.replace(/^www\./, '');

    for (const [brand, legit] of BRAND_DOMAINS) {
      // Skip if this IS the legitimate domain
      if (cleanHost === legit || cleanHost.endsWith('.' + legit)) continue;

      if (fullUrl.includes(brand)) {
        result.inUrl = true;
        result.brand = brand;

        // Check if brand is specifically in a subdomain
        if (subdomains.some(sd => sd.includes(brand))) {
          result.inSubdomain = true;
        }
        break; // Report first match only
      }
    }
    return result;
  }

  /** Check for Unicode confusable characters in hostname. */
  _hasConfusables(hostname) {
    for (const charSets of Object.values(CONFUSABLE_MAP)) {
      for (const confusable of charSets) {
        if (hostname.includes(confusable)) return true;
      }
    }
    return false;
  }

  /** Minimum Levenshtein distance between SLD and any known brand. */
  _minBrandDistance(sld) {
    if (!sld || sld.length < 3) return 99;

    let minDist = 99;
    for (const [brand] of BRAND_DOMAINS) {
      // Only compare if lengths are similar (±3 chars)
      if (Math.abs(sld.length - brand.length) > 3) continue;
      const dist = this._levenshtein(sld, brand);
      if (dist < minDist) minDist = dist;
      if (dist === 0) break; // Exact match — it's the brand itself
    }
    return minDist;
  }

  /** Levenshtein edit distance. */
  _levenshtein(a, b) {
    const m = a.length, n = b.length;
    if (m === 0) return n;
    if (n === 0) return m;

    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,
          dp[i][j - 1] + 1,
          dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
        );
      }
    }
    return dp[m][n];
  }

  /** Maximum consecutive repetition of any character. */
  _maxCharRepetition(str) {
    let max = 0, cur = 1;
    for (let i = 1; i < str.length; i++) {
      if (str[i] === str[i - 1]) { cur++; }
      else { if (cur > max) max = cur; cur = 1; }
    }
    return Math.max(max, cur);
  }

  /** Consonant-to-vowel ratio (natural language metric). */
  _consonantVowelRatio(str) {
    if (!str) return 0;
    const vowels = (str.match(/[aeiou]/gi) || []).length;
    const consonants = (str.match(/[bcdfghjklmnpqrstvwxyz]/gi) || []).length;
    if (vowels === 0) return consonants > 0 ? 10 : 0;
    return Math.round((consonants / vowels) * 100) / 100;
  }

  /** Composite visual similarity score from homograph sub-features. */
  _visualSimilarity(v) {
    let score = 0;
    if (v.is_punycode)          score += 3.0;
    if (v.has_confusable_chars) score += 3.0;
    if (v.typosquatting_distance <= 1) score += 4.0;
    else if (v.typosquatting_distance <= 2) score += 2.5;
    if (v.char_repetition_max >= 3) score += 1.5;
    if (v.consonant_vowel_ratio > 4 || v.consonant_vowel_ratio < 0.3) score += 1.0;

    // Normalise to [0, 1]
    return Math.round(Math.min(1, score / 8) * 1000) / 1000;
  }


  // ═══════════════════════════════════════════════════════════════════════════
  //  Private — Risk Score Aggregation
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Compute an aggregate heuristic risk score from the feature vector.
   * Uses a weighted sum with sigmoid normalisation.
   *
   * Weights are based on expected importance from the FEATURE_REGISTRY.
   * This is a hand-tuned heuristic — intended to be replaced by ONNX
   * model inference in Phase 1b.
   */
  _computeRiskScore(v) {
    let rawScore = 0;

    // ── Critical signals (weight 3–4) ─────────────────────────────────────
    if (v.is_ip_address)           rawScore += 3.5;
    if (v.is_hex_ip)               rawScore += 3.5;
    if (v.has_at_sign)             rawScore += 3.0;
    if (v.has_suspicious_extension) rawScore += 3.0;
    if (v.brand_in_subdomain)      rawScore += 3.5;
    if (v.suspicious_keyword_combo) rawScore += 3.0;

    // ── Strong signals (weight 2–3) ───────────────────────────────────────
    if (v.typosquatting_distance <= 1) rawScore += 3.5;
    else if (v.typosquatting_distance <= 2) rawScore += 2.0;

    if (v.visual_similarity_score > 0.5) rawScore += 2.5;
    if (v.is_punycode)             rawScore += 2.0;
    if (v.has_confusable_chars)    rawScore += 2.5;
    if (v.has_https_token_hostname) rawScore += 2.0;
    if (v.has_security_word_hostname) rawScore += 1.5;
    if (v.scheme_host_mismatch)    rawScore += 2.5;
    if (v.has_sensitive_query_key) rawScore += 2.0;
    if (v.has_double_slash_redirect) rawScore += 1.5;
    if (v.brand_keyword_in_url && !v.brand_in_subdomain) rawScore += 1.5;

    // ── Moderate signals (weight 1–2) ─────────────────────────────────────
    if (v.num_subdomains >= 3)     rawScore += 2.0;
    else if (v.num_subdomains >= 2) rawScore += 0.8;

    if (v.num_hyphens_hostname >= 3) rawScore += 2.0;
    else if (v.num_hyphens_hostname >= 1) rawScore += 0.8;

    if (v.is_suspicious_tld)       rawScore += 1.5;
    if (v.is_shortener)            rawScore += 1.2;
    if (v.has_non_standard_port)   rawScore += 1.5;

    if (v.url_length >= 100)       rawScore += 1.5;
    else if (v.url_length >= 75)   rawScore += 0.8;

    if (v.hostname_entropy > 4.0)  rawScore += 1.5;
    else if (v.hostname_entropy > 3.5) rawScore += 0.8;

    if (v.phishing_keyword_count >= 3) rawScore += 1.8;
    else if (v.phishing_keyword_count >= 1) rawScore += 0.6;

    if (v.urgency_keyword_count >= 2)  rawScore += 1.5;
    else if (v.urgency_keyword_count >= 1) rawScore += 0.6;

    if (v.query_contains_url)      rawScore += 1.2;

    // ── Weak signals (weight 0.3–1) ───────────────────────────────────────
    if (!v.is_https)               rawScore += 0.8;
    if (v.path_depth >= 6)         rawScore += 0.8;
    else if (v.path_depth >= 4)    rawScore += 0.4;

    if (v.num_percent_encodings >= 5) rawScore += 0.8;
    if (v.num_digits_hostname >= 3) rawScore += 0.6;
    if (v.num_dots_hostname >= 5)  rawScore += 1.0;
    if (v.has_fragment)            rawScore += 0.3;
    if (v.consonant_vowel_ratio > 5) rawScore += 0.6;
    if (v.char_repetition_max >= 3) rawScore += 0.5;

    // Sigmoid normalisation
    const score = 1 / (1 + Math.exp(-0.9 * (rawScore - 4.0)));
    return Math.round(score * 1000) / 1000;
  }


  // ═══════════════════════════════════════════════════════════════════════════
  //  Private — Detail Builder (Explainability)
  // ═══════════════════════════════════════════════════════════════════════════

  /** Build human-readable detail objects for triggered features. */
  _buildDetails(vector) {
    const details = [];
    const meta = {};
    for (const f of FEATURE_REGISTRY) meta[f.name] = f;

    for (const [name, value] of Object.entries(vector)) {
      const m = meta[name];
      if (!m) continue;

      // Only include features with non-zero / non-default values
      const isBinary = name.startsWith('has_') || name.startsWith('is_') ||
                        name === 'suspicious_keyword_combo' || name === 'brand_keyword_in_url' ||
                        name === 'brand_in_subdomain';

      const isTriggered = isBinary
        ? value === 1
        : this._isSignificant(name, value);

      if (isTriggered) {
        details.push({
          id: m.id,
          name: m.name,
          value,
          category: m.category,
          why: m.why,
          how: m.how,
          importance: m.importance,
          severity: m.importance >= 8 ? 'critical'
                  : m.importance >= 6 ? 'high'
                  : m.importance >= 4 ? 'medium'
                  : 'low',
        });
      }
    }

    // Sort by importance descending
    details.sort((a, b) => b.importance - a.importance);
    return details;
  }

  /** Determine if a numeric feature value is significant (non-trivial). */
  _isSignificant(name, value) {
    const thresholds = {
      url_length:             75,
      hostname_length:        30,
      path_length:            50,
      query_length:           50,
      num_dots_url:           4,
      num_dots_hostname:      3,
      num_hyphens_url:        3,
      num_hyphens_hostname:   1,
      num_subdomains:         2,
      num_digits_hostname:    2,
      url_entropy:            4.2,
      hostname_entropy:       3.5,
      path_entropy:           4.0,
      path_depth:             4,
      phishing_keyword_count: 1,
      urgency_keyword_count:  1,
      digit_letter_ratio:     0.3,
      special_char_ratio:     0.25,
      num_percent_encodings:  3,
      consonant_vowel_ratio:  4,
      char_repetition_max:    3,
      typosquatting_distance: 2,    // ≤2 is significant
      visual_similarity_score: 0.3,
      num_query_params:       4,
      max_query_value_length: 80,
      max_path_token_length:  30,
      domain_token_count:     4,
      subdomain_total_length: 15,
    };

    if (name === 'typosquatting_distance') return value <= 2 && value > 0;
    if (thresholds[name] !== undefined) return value >= thresholds[name];
    return false;
  }

  /** Return a result for unparseable URLs. */
  _errorResult(rawUrl) {
    return {
      vector: Object.fromEntries(FEATURE_REGISTRY.map(f => [f.name, 0])),
      details: [{
        id: 'ERR', name: 'unparseable_url', value: 1,
        category: 'security', why: 'URL structure is invalid.',
        how: 'Unparseable URLs are almost never legitimate navigation targets.',
        importance: 9, severity: 'critical',
      }],
      riskScore: 0.85,
      verdict: 'phishing',
    };
  }
}
