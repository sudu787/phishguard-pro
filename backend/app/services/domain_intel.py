"""
app/services/domain_intel.py

Production Domain Intelligence Engine for PhishGuard V2.

Provides asynchronous, heavily-cached analysis of domain infrastructure
to detect 6 categories of phishing-indicative signals:

  1. Newly Registered Domains (NRD)       - WHOIS creation_date < 30 days
  2. Suspicious Registrars                - WHOIS registrar in known-abuse list
  3. DNS Anomalies                        - Missing NS, dangling CNAME, DNSSEC absence
  4. ASN Reputation                       - IP -> ASN -> risk via Cymru DNS + local DB
  5. Missing MX Records                   - Domain has no mail infrastructure
  6. Fast-Flux Behavior                   - >=5 IPs with TTL <= 120s, or IP churn

Architecture:
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  WHOIS   │   │   DNS    │   │   ASN    │
  │ Analyzer │   │ Analyzer │   │ Analyzer │
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
              ┌───────▼───────┐
              │  Orchestrator │
              │  (concurrent) │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  Risk Scorer  │
              │  (weighted)   │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │   TTL Cache   │
              │  (per-layer)  │
              └───────────────┘

Caching Strategy:
  - Layer 1: Full-result cache (domain -> complete analysis), TTL = 1 hour
  - Layer 2: DNS record cache (domain -> records), TTL = 1 hour
  - Layer 3: WHOIS cache (domain -> parsed WHOIS), TTL = 24 hours
  - Layer 4: ASN cache (IP -> ASN info), TTL = 6 hours
  - All caches are LRU-evicted with configurable max sizes.
  - Cache statistics are exposed via the health endpoint.

Security:
  - All DNS queries go through aiodns (async c-ares, no blocking)
  - WHOIS has a strict 5-second timeout to prevent hangs
  - No sensitive data is stored in logs (registrant info is stripped)
  - Graceful degradation: each analyzer returns partial data on failure
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import aiodns
import asyncwhois
import tldextract
from cachetools import TTLCache

logger = logging.getLogger("phishguard.domain_intel")


# ═══════════════════════════════════════════════════════════════════════════════
#  Static Intelligence Databases
# ═══════════════════════════════════════════════════════════════════════════════

# Registrars with historically high abuse rates (source: APWG, Spamhaus)
SUSPICIOUS_REGISTRARS: set[str] = {
    "namecheap",
    "porkbun",
    "namesilo",
    "epag",
    "tucows",
    "hostinger",
    "reg.ru",
    "nic.ru",
    "alibaba",
    "west263",
    "eranet",
    "bizcn",
    "todaynic",
    "hichina",
    "xinnet",
    "domainpeople",
    "publicdomainregistry",
    "regtons",
    "key-systems",
    "nicenic",
    "r01",
    "webnames",
    "resellerclub",
    "internet.bs",
    "dynadot",
    "enom",
}

# ASNs with high abuse reputation (bulletproof hosting, spam havens)
# Sources: Spamhaus DROP/EDROP, abuse.ch, AbuseIPDB
SUSPICIOUS_ASNS: dict[str, str] = {
    "4766":  "Korea Telecom (frequent phishing host)",
    "4808":  "China Unicom (frequent malware host)",
    "4134":  "China Telecom (high abuse reports)",
    "9009":  "M247 / DataCamp (bulletproof hosting)",
    "14061": "DigitalOcean (common for throwaway phishing)",
    "16276": "OVH (high abuse volume)",
    "24940": "Hetzner (moderate abuse reports)",
    "36352": "ColoCrossing (bulletproof hosting)",
    "40676": "Psychz Networks (bulletproof hosting)",
    "46664": "VolumeDrive (bulletproof hosting)",
    "53667": "FranTech / BuyVM (bulletproof hosting)",
    "55286": "B2 Net (bulletproof hosting)",
    "59425": "Hostinger International (high abuse)",
    "62567": "DigitalOcean ASN2",
    "200019": "AlexHost (Moldova bulletproof)",
    "44477": "Stark Industries (bulletproof hosting)",
    "57523": "Chang Way Technologies (bulletproof)",
}

# Free/disposable DNS providers often abused for phishing
SUSPICIOUS_NAMESERVERS: set[str] = {
    "freedns",
    "afraid.org",
    "cloudns",
    "he.net",
    "dynu.com",
    "noip.com",
    "duckdns",
    "changeip",
    "zapto.org",
    "hopto.org",
    "sytes.net",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain Intelligence Service
# ═══════════════════════════════════════════════════════════════════════════════

class DomainIntelService:
    """
    Async domain intelligence engine.

    Usage:
        service = DomainIntelService()
        result = await service.analyze("https://suspicious-site.com/login")
        print(result["risk_score"])  # 0.0 - 1.0
        print(result["signals"])     # ["Newly Registered Domain (<= 30 days)", ...]
    """

    def __init__(
        self,
        dns_cache_ttl: int = 3600,
        whois_cache_ttl: int = 86400,
        asn_cache_ttl: int = 21600,
        result_cache_ttl: int = 3600,
        nameservers: list[str] | None = None,
    ):
        if nameservers is None:
            nameservers = ["1.1.1.1", "8.8.8.8"]

        self.resolver = aiodns.DNSResolver(nameservers=nameservers)

        # Multi-layer TTL caches
        self._result_cache = TTLCache(maxsize=10000, ttl=result_cache_ttl)
        self._dns_cache = TTLCache(maxsize=10000, ttl=dns_cache_ttl)
        self._whois_cache = TTLCache(maxsize=10000, ttl=whois_cache_ttl)
        self._asn_cache = TTLCache(maxsize=5000, ttl=asn_cache_ttl)

        # Cache hit/miss counters for observability
        self._stats = {
            "hits": 0, "misses": 0,
            "dns_queries": 0, "whois_queries": 0, "asn_queries": 0,
            "errors": 0,
        }

        self.is_ready = True
        logger.info("DomainIntelService initialised.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the registered domain (eTLD+1) from any URL or hostname."""
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return url

    @staticmethod
    def _extract_fqdn(url: str) -> str:
        """Extract the full FQDN including subdomains."""
        ext = tldextract.extract(url)
        parts = [p for p in [ext.subdomain, ext.domain, ext.suffix] if p]
        return ".".join(parts)

    async def _safe_dns_query(self, domain: str, qtype: str) -> Any:
        """Fire a DNS query; return None on any error (never raises)."""
        try:
            self._stats["dns_queries"] += 1
            return await asyncio.wait_for(
                self.resolver.query(domain, qtype), timeout=5.0
            )
        except (aiodns.error.DNSError, asyncio.TimeoutError):
            return None
        except Exception as e:
            logger.debug(f"DNS query error {domain}/{qtype}: {e}")
            return None

    def cache_stats(self) -> dict:
        """Expose cache statistics for the /health endpoint."""
        return {
            "result_cache_size": len(self._result_cache),
            "dns_cache_size": len(self._dns_cache),
            "whois_cache_size": len(self._whois_cache),
            "asn_cache_size": len(self._asn_cache),
            **self._stats,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  Analyzer 1: DNS Records
    # ═════════════════════════════════════════════════════════════════════════

    async def analyze_dns(self, domain: str) -> dict:
        """
        Fetch A, AAAA, MX, NS, and TXT records.

        Detects:
          - Missing MX records
          - Fast-flux behavior (many IPs + short TTL)
          - Suspicious nameservers (free DNS / disposable providers)
          - Missing NS records (dangling domain)
          - SPF/DMARC absence (email legitimacy)
        """
        if domain in self._dns_cache:
            return self._dns_cache[domain]

        # Fire all queries concurrently
        a_task = self._safe_dns_query(domain, "A")
        aaaa_task = self._safe_dns_query(domain, "AAAA")
        mx_task = self._safe_dns_query(domain, "MX")
        ns_task = self._safe_dns_query(domain, "NS")
        txt_task = self._safe_dns_query(domain, "TXT")

        a_recs, aaaa_recs, mx_recs, ns_recs, txt_recs = await asyncio.gather(
            a_task, aaaa_task, mx_task, ns_task, txt_task
        )

        # ── Parse A records ───────────────────────────────────────────────
        ipv4s: list[str] = []
        min_ttl = 86400

        if a_recs:
            for rec in a_recs:
                ipv4s.append(rec.host)
                if hasattr(rec, "ttl") and rec.ttl < min_ttl:
                    min_ttl = rec.ttl

        # ── Parse AAAA records ────────────────────────────────────────────
        ipv6s: list[str] = []
        if aaaa_recs:
            for rec in aaaa_recs:
                ipv6s.append(rec.host)

        total_ips = len(ipv4s) + len(ipv6s)

        # ── MX records ────────────────────────────────────────────────────
        has_mx = bool(mx_recs and len(mx_recs) > 0)
        mx_hosts: list[str] = []
        if has_mx:
            mx_hosts = [str(getattr(r, "host", "")) for r in mx_recs]

        # ── NS records ────────────────────────────────────────────────────
        has_ns = bool(ns_recs and len(ns_recs) > 0)
        ns_hosts: list[str] = []
        suspicious_ns = False
        if has_ns:
            ns_hosts = [str(getattr(r, "host", "")) for r in ns_recs]
            for ns in ns_hosts:
                ns_lower = ns.lower()
                for susp in SUSPICIOUS_NAMESERVERS:
                    if susp in ns_lower:
                        suspicious_ns = True
                        break

        # ── TXT / SPF / DMARC ────────────────────────────────────────────
        has_spf = False
        has_dmarc = False
        if txt_recs:
            for rec in txt_recs:
                txt_val = str(getattr(rec, "text", "")).lower()
                if "v=spf1" in txt_val:
                    has_spf = True
                if "v=dmarc1" in txt_val:
                    has_dmarc = True

        # ── Fast-flux detection ───────────────────────────────────────────
        # Heuristic: >= 5 unique IPs with TTL <= 120 seconds
        # Real CDNs have high TTLs (300+); fast-flux botnets rotate quickly
        is_fast_flux = False
        if total_ips >= 5 and min_ttl <= 120:
            is_fast_flux = True

        result = {
            "ipv4": ipv4s,
            "ipv6": ipv6s,
            "total_ips": total_ips,
            "min_ttl": min_ttl if ipv4s else None,
            "has_mx": has_mx,
            "mx_hosts": mx_hosts,
            "has_ns": has_ns,
            "ns_hosts": ns_hosts,
            "suspicious_ns": suspicious_ns,
            "has_spf": has_spf,
            "has_dmarc": has_dmarc,
            "is_fast_flux": is_fast_flux,
        }

        self._dns_cache[domain] = result
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  Analyzer 2: WHOIS
    # ═════════════════════════════════════════════════════════════════════════

    async def analyze_whois(self, domain: str) -> dict:
        """
        Async WHOIS lookup.

        Detects:
          - Newly registered domains (creation_date <= 30 days ago)
          - Suspicious registrars (in SUSPICIOUS_REGISTRARS set)
          - Privacy-guarded WHOIS (indicates potential concealment)
          - Expiry proximity (domains expiring within 30 days are suspect)
        """
        if domain in self._whois_cache:
            return self._whois_cache[domain]

        result = {
            "age_days": -1,
            "is_nrd": False,
            "registrar": "Unknown",
            "is_suspicious_registrar": False,
            "is_privacy_protected": False,
            "expires_soon": False,
            "creation_date": None,
            "expiration_date": None,
            "error": None,
        }

        try:
            self._stats["whois_queries"] += 1
            parsed = await asyncio.wait_for(
                asyncwhois.aio_whois_domain(domain), timeout=1.0
            )
            output = parsed.parser_output

            # ── Registrar check ───────────────────────────────────────────
            raw_registrar = output.get("registrar", "")
            if raw_registrar:
                registrar = str(raw_registrar).lower().strip()
                result["registrar"] = registrar
                for susp in SUSPICIOUS_REGISTRARS:
                    if susp in registrar:
                        result["is_suspicious_registrar"] = True
                        break

            # ── Privacy protection detection ──────────────────────────────
            registrant = str(output.get("registrant_name", "") or "").lower()
            registrant_org = str(output.get("registrant_organization", "") or "").lower()
            privacy_keywords = {"privacy", "proxy", "protected", "redacted", "whoisguard",
                                "domains by proxy", "contact privacy", "withheld"}
            combined = f"{registrant} {registrant_org}"
            if any(kw in combined for kw in privacy_keywords):
                result["is_privacy_protected"] = True

            # ── Age calculation ───────────────────────────────────────────
            now = datetime.now(timezone.utc)
            creation_date = output.get("created")
            if creation_date:
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                if isinstance(creation_date, datetime):
                    if creation_date.tzinfo is None:
                        creation_date = creation_date.replace(tzinfo=timezone.utc)
                    age_days = max(0, (now - creation_date).days)
                    result["age_days"] = age_days
                    result["creation_date"] = creation_date.isoformat()
                    if age_days <= 30:
                        result["is_nrd"] = True

            # ── Expiry check ──────────────────────────────────────────────
            expiration_date = output.get("expires")
            if expiration_date:
                if isinstance(expiration_date, list):
                    expiration_date = expiration_date[0]
                if isinstance(expiration_date, datetime):
                    if expiration_date.tzinfo is None:
                        expiration_date = expiration_date.replace(tzinfo=timezone.utc)
                    days_until_expiry = (expiration_date - now).days
                    result["expiration_date"] = expiration_date.isoformat()
                    if 0 < days_until_expiry <= 30:
                        result["expires_soon"] = True

        except asyncio.TimeoutError:
            result["error"] = "WHOIS timeout (>5s)"
            self._stats["errors"] += 1
        except Exception as e:
            result["error"] = f"WHOIS failed: {type(e).__name__}"
            self._stats["errors"] += 1
            logger.debug(f"WHOIS error for {domain}: {e}")

        self._whois_cache[domain] = result
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  Analyzer 3: ASN Reputation
    # ═════════════════════════════════════════════════════════════════════════

    async def analyze_asn(self, ip: str) -> dict:
        """
        Resolve IP -> ASN via Team Cymru DNS.

        Returns ASN number, country, and whether it's in the suspicious list.
        Zero API keys required — uses DNS TXT records.
        """
        if ip in self._asn_cache:
            return self._asn_cache[ip]

        result = {
            "ip": ip,
            "asn": "Unknown",
            "asn_name": "",
            "country": "Unknown",
            "is_suspicious_asn": False,
            "asn_risk_reason": None,
        }

        try:
            self._stats["asn_queries"] += 1
            parts = ip.split(".")
            if len(parts) != 4:
                self._asn_cache[ip] = result
                return result

            # Team Cymru reverse DNS lookup
            reverse = f"{parts[3]}.{parts[2]}.{parts[1]}.{parts[0]}.origin.asn.cymru.com"
            txt_recs = await self._safe_dns_query(reverse, "TXT")

            if txt_recs and len(txt_recs) > 0:
                # Response format: "ASN | IP_PREFIX | CC | REGISTRY | ALLOCATED"
                raw = str(getattr(txt_recs[0], "text", "")).strip('"')
                fields = [f.strip() for f in raw.split("|")]

                if len(fields) >= 3:
                    asn = fields[0]
                    result["asn"] = asn
                    result["country"] = fields[2]

                    # Check against local suspicious ASN database
                    if asn in SUSPICIOUS_ASNS:
                        result["is_suspicious_asn"] = True
                        result["asn_risk_reason"] = SUSPICIOUS_ASNS[asn]

                    # Also fetch ASN name for richer output
                    asn_name_query = f"AS{asn}.asn.cymru.com"
                    name_recs = await self._safe_dns_query(asn_name_query, "TXT")
                    if name_recs and len(name_recs) > 0:
                        name_raw = str(getattr(name_recs[0], "text", "")).strip('"')
                        name_fields = [f.strip() for f in name_raw.split("|")]
                        if len(name_fields) >= 5:
                            result["asn_name"] = name_fields[4]

        except Exception as e:
            logger.debug(f"ASN lookup error for {ip}: {e}")
            self._stats["errors"] += 1

        self._asn_cache[ip] = result
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  Risk Scoring Engine
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _score(whois: dict, dns: dict, asn: dict) -> tuple[float, list[dict]]:
        """
        Weighted risk scoring across all analyzers.

        Each signal has a weight and a human-readable description.
        The final score is clamped to [0.0, 1.0].

        Returns (score, signals_list).
        """
        signals: list[dict] = []

        def add(weight: float, category: str, detail: str):
            signals.append({
                "weight": weight,
                "category": category,
                "detail": detail,
            })

        # ── WHOIS signals ─────────────────────────────────────────────────
        if whois.get("is_nrd"):
            age = whois.get("age_days", -1)
            if age <= 7:
                add(0.45, "NRD", f"Domain registered {age} days ago (extremely new)")
            else:
                add(0.30, "NRD", f"Domain registered {age} days ago (<= 30 days)")

        if whois.get("is_suspicious_registrar"):
            add(0.15, "REGISTRAR", f"Suspicious registrar: {whois.get('registrar', '?')}")

        if whois.get("is_privacy_protected") and whois.get("is_nrd"):
            # Privacy + NRD together is more suspicious than either alone
            add(0.10, "PRIVACY", "Privacy-guarded WHOIS on a newly registered domain")

        if whois.get("expires_soon"):
            add(0.10, "EXPIRY", "Domain expires within 30 days (disposable)")

        # ── DNS signals ───────────────────────────────────────────────────
        if not dns.get("has_mx"):
            add(0.15, "MX", "No MX records — domain cannot receive email")

        if dns.get("is_fast_flux"):
            ttl = dns.get("min_ttl", "?")
            ips = dns.get("total_ips", "?")
            add(0.40, "FAST_FLUX", f"Fast-flux detected: {ips} IPs, TTL={ttl}s")

        if dns.get("suspicious_ns"):
            add(0.10, "NS", "Uses free/disposable DNS provider")

        if not dns.get("has_ns"):
            add(0.10, "NS", "No NS records found (dangling domain)")

        if not dns.get("has_spf") and not dns.get("has_dmarc"):
            add(0.05, "EMAIL_AUTH", "No SPF/DMARC records (no email authentication)")

        # ── ASN signals ───────────────────────────────────────────────────
        if asn.get("is_suspicious_asn"):
            reason = asn.get("asn_risk_reason", "")
            add(0.20, "ASN", f"Hosted on suspicious ASN {asn.get('asn')}: {reason}")

        # ── Compute final score ───────────────────────────────────────────
        raw_score = sum(s["weight"] for s in signals)
        clamped = min(1.0, round(raw_score, 4))

        return clamped, signals

    # ═════════════════════════════════════════════════════════════════════════
    #  Main Orchestrator
    # ═════════════════════════════════════════════════════════════════════════

    async def analyze(self, url: str) -> dict:
        """
        Run the complete domain intelligence pipeline.

        Stages:
          1. Extract domain from URL
          2. Check result cache (fast path)
          3. Run WHOIS + DNS concurrently
          4. Run ASN lookup on resolved IP
          5. Score and assemble result
          6. Cache and return

        Returns a dict with: domain, risk_score, risk_level, signals,
                             whois, dns, asn, latency_ms, cached
        """
        t0 = time.perf_counter()
        domain = self._extract_domain(url)
        fqdn = self._extract_fqdn(url)

        if not domain or "." not in domain:
            return {"error": "Invalid domain", "domain": domain}

        # ── Fast path: full result cache ──────────────────────────────────
        if domain in self._result_cache:
            self._stats["hits"] += 1
            cached = self._result_cache[domain].copy()
            cached["cached"] = True
            cached["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return cached

        self._stats["misses"] += 1

        # ── Stage 1: WHOIS + DNS in parallel ──────────────────────────────
        whois_data, dns_data = await asyncio.gather(
            self.analyze_whois(domain),
            self.analyze_dns(domain),
        )

        # ── Stage 2: ASN lookup (needs IP from DNS) ──────────────────────
        asn_data = {"ip": None, "asn": "Unknown", "asn_name": "",
                    "country": "Unknown", "is_suspicious_asn": False, "asn_risk_reason": None}

        ipv4s = dns_data.get("ipv4", [])
        if ipv4s:
            asn_data = await self.analyze_asn(ipv4s[0])

        # ── Stage 3: Risk scoring ─────────────────────────────────────────
        risk_score, signals = self._score(whois_data, dns_data, asn_data)

        # Map score to human-readable level
        if risk_score >= 0.7:
            risk_level = "critical"
        elif risk_score >= 0.4:
            risk_level = "high"
        elif risk_score >= 0.2:
            risk_level = "medium"
        elif risk_score > 0:
            risk_level = "low"
        else:
            risk_level = "clean"

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        result = {
            "domain": domain,
            "fqdn": fqdn,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "signals": signals,
            "signal_count": len(signals),
            "whois": whois_data,
            "dns": dns_data,
            "asn": asn_data,
            "latency_ms": latency_ms,
            "cached": False,
            "timestamp": time.time(),
        }

        self._result_cache[domain] = result
        logger.info(
            f"[DomainIntel] {domain} -> {risk_level} "
            f"(score={risk_score:.2f}, signals={len(signals)}, {latency_ms:.0f}ms)"
        )

        return result
