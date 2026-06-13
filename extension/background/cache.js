/**
 * PhishGuard V2 — Session Cache Manager
 * background/cache.js
 *
 * Provides a TTL-based cache backed by chrome.storage.session.
 * Session storage is automatically cleared when the browser closes,
 * preventing stale results from persisting across sessions.
 *
 * Key design decisions:
 *   - Keys are URL hashes (Base64-encoded, truncated) for privacy
 *   - TTL is enforced on read (lazy expiration)
 *   - Max cache size is bounded to prevent storage quota exhaustion
 */

const CACHE_PREFIX = 'pg_';
const DEFAULT_TTL_MS = 5 * 60 * 1000; // 5 minutes
const MAX_CACHE_ENTRIES = 200;


export class CacheManager {

  /**
   * @param {number} [ttlMs=300000] — Time-to-live in milliseconds
   */
  constructor(ttlMs = DEFAULT_TTL_MS) {
    this._ttl = ttlMs;
  }

  /**
   * Retrieve a cached scan result for a URL.
   * Returns null if not found or expired.
   *
   * @param  {string} url
   * @return {Promise<Object|null>}
   */
  async get(url) {
    const key = this._makeKey(url);
    try {
      const data = await chrome.storage.session.get(key);
      const entry = data[key];
      if (!entry) return null;

      // Enforce TTL
      if (Date.now() - entry.timestamp > this._ttl) {
        await chrome.storage.session.remove(key);
        return null;
      }

      return entry;
    } catch (err) {
      console.warn('[PhishGuard Cache] Read error:', err.message);
      return null;
    }
  }

  /**
   * Cache a scan result.
   *
   * @param {string} url
   * @param {Object} result — The ScanResult object
   */
  async set(url, result) {
    const key = this._makeKey(url);
    try {
      // Enforce max cache size (evict oldest if full)
      await this._evictIfNeeded();
      await chrome.storage.session.set({ [key]: result });
    } catch (err) {
      console.warn('[PhishGuard Cache] Write error:', err.message);
    }
  }

  /**
   * Store per-tab scan result (used by popup to retrieve current tab's result).
   *
   * @param {number} tabId
   * @param {Object} result
   */
  async setTabResult(tabId, result) {
    try {
      await chrome.storage.session.set({ [`tab_${tabId}`]: result });
    } catch (err) {
      console.warn('[PhishGuard Cache] Tab result write error:', err.message);
    }
  }

  /**
   * Retrieve the scan result for a specific tab.
   *
   * @param  {number} tabId
   * @return {Promise<Object|null>}
   */
  async getTabResult(tabId) {
    try {
      const data = await chrome.storage.session.get(`tab_${tabId}`);
      return data[`tab_${tabId}`] || null;
    } catch {
      return null;
    }
  }

  /**
   * Clear all cached entries. Used for debugging / reset.
   */
  async clear() {
    try {
      await chrome.storage.session.clear();
    } catch (err) {
      console.warn('[PhishGuard Cache] Clear error:', err.message);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Private helpers
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Generate a stable, privacy-preserving cache key from a URL.
   * Uses a simple hash to avoid storing full URLs in session storage.
   */
  _makeKey(url) {
    let hash = 0;
    for (let i = 0; i < url.length; i++) {
      const chr = url.charCodeAt(i);
      hash = ((hash << 5) - hash) + chr;
      hash |= 0; // Convert to 32-bit integer
    }
    return CACHE_PREFIX + Math.abs(hash).toString(36);
  }

  /**
   * Evict oldest cache entries if we exceed MAX_CACHE_ENTRIES.
   * Prevents unbounded storage growth during long browsing sessions.
   */
  async _evictIfNeeded() {
    try {
      const all = await chrome.storage.session.get(null);
      const cacheKeys = Object.keys(all).filter(k => k.startsWith(CACHE_PREFIX));

      if (cacheKeys.length < MAX_CACHE_ENTRIES) return;

      // Sort by timestamp (oldest first) and remove the oldest 20%
      const entries = cacheKeys
        .map(k => ({ key: k, ts: all[k]?.timestamp || 0 }))
        .sort((a, b) => a.ts - b.ts);

      const evictCount = Math.ceil(cacheKeys.length * 0.2);
      const keysToRemove = entries.slice(0, evictCount).map(e => e.key);

      await chrome.storage.session.remove(keysToRemove);
    } catch {
      // Non-critical — don't block the scan pipeline on cache maintenance
    }
  }
}
