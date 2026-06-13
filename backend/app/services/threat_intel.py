"""
app/services/threat_intel.py

Asynchronous threat-intelligence service.
Checks URLs against:
  1. Local in-memory blocklist cache refreshed from OpenPhish feed (~3000 URLs)
  2. URLHaus API (per-URL async lookup)

The blocklist is refreshed on a background task every CACHE_TTL seconds.
All network I/O is async (httpx) — never blocks the FastAPI event loop.
"""

import asyncio
import logging
import time
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("phishguard.intel")


class ThreatIntelService:
    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        openphish_url: str = "https://openphish.com/feed.txt",
        urlhaus_api: str = "https://urlhaus-api.abuse.ch/v1/url/",
    ):
        self._cache_ttl = cache_ttl_seconds
        self._openphish_url = openphish_url
        self._urlhaus_api = urlhaus_api
        self._blocklist: set[str] = set()
        self._last_refresh: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None
        self._refresh_lock = asyncio.Lock()
        self.is_ready = False

    async def warm_up(self):
        """Called at startup — pre-loads blocklist and creates HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=None),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        await self._refresh_blocklist()
        self.is_ready = True

    async def close(self):
        if self._client:
            await self._client.aclose()

    # ── Public interface ──────────────────────────────────────────────────────
    async def check(self, url: str) -> dict:
        """
        Check a URL against all threat sources.
        Returns: {"in_blocklist": bool, "sources": list[str]}
        """
        # Refresh stale cache (non-blocking — uses lock to avoid stampede)
        if time.time() - self._last_refresh > self._cache_ttl:
            asyncio.create_task(self._maybe_refresh())

        sources = []

        # Fast local check against OpenPhish blocklist
        normalised = _normalise_url(url)
        if normalised in self._blocklist:
            sources.append("OpenPhish")

        # Also check domain-only match (catches paths we might not have)
        domain = _extract_domain(url)
        domain_matches = [u for u in self._blocklist if domain and domain in u]
        if domain_matches:
            if "OpenPhish" not in sources:
                sources.append("OpenPhish (domain match)")

        # URLHaus per-URL lookup (async, returns quickly from their cache)
        urlhaus_result = await self._check_urlhaus(url)
        if urlhaus_result:
            sources.append("URLHaus")

        return {
            "in_blocklist": len(sources) > 0,
            "sources": sources,
        }

    # ── Blocklist refresh ─────────────────────────────────────────────────────
    async def _maybe_refresh(self):
        """Refresh only if not already being refreshed (lock guard)."""
        if self._refresh_lock.locked():
            return
        async with self._refresh_lock:
            if time.time() - self._last_refresh > self._cache_ttl:
                await self._refresh_blocklist()

    async def _refresh_blocklist(self):
        """Download and parse the OpenPhish feed into the in-memory set."""
        try:
            resp = await self._client.get(self._openphish_url)
            resp.raise_for_status()
            new_list = set()
            for line in resp.text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    new_list.add(_normalise_url(line))
            self._blocklist = new_list
            self._last_refresh = time.time()
            logger.info(f"OpenPhish blocklist refreshed: {len(self._blocklist)} entries")
        except Exception as exc:
            logger.warning(f"Failed to refresh OpenPhish blocklist: {exc}")
            # Keep stale cache — don't clear it on failure

    # ── URLHaus lookup ────────────────────────────────────────────────────────
    async def _check_urlhaus(self, url: str) -> bool:
        """
        Query URLHaus API for a single URL.
        Returns True if the URL is listed as active/suspicious.
        Degrades gracefully on timeout or error.
        """
        try:
            resp = await self._client.post(
                self._urlhaus_api,
                data={"url": url},
                timeout=httpx.Timeout(3.0),
            )
            if resp.status_code == 200:
                data = resp.json()
                query_status = data.get("query_status", "")
                url_status = data.get("url_status", "")
                return query_status == "is_listed" and url_status in ("online", "unknown")
        except (httpx.TimeoutException, httpx.RequestError):
            pass  # Degrade silently — intel is best-effort
        except Exception as exc:
            logger.debug(f"URLHaus check error: {exc}")
        return False


# ─── Utility functions ────────────────────────────────────────────────────────
def _normalise_url(url: str) -> str:
    """Strip trailing slashes and lowercase scheme/host for consistent matching."""
    url = url.strip().lower().rstrip("/")
    return url


def _extract_domain(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        return parsed.hostname
    except Exception:
        return None
