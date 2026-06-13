/**
 * PhishGuard Pro — background.js (Manifest V3 Service Worker)
 *
 * Orchestrates the 3-tier detection pipeline:
 *   Tier 1 → Fast URL analysis via backend /predict/url   (~5–20ms)
 *   Tier 2 → DOM features injected from content.js        (~30–80ms)
 *   Tier 3 → Async threat-intel check via /predict/intel  (non-blocking)
 *
 * Design principles:
 *  - Never blocks the UI thread (all work is async, off main thread).
 *  - Per-tab debouncing prevents duplicate requests on rapid navigation.
 *  - Results are cached in chrome.storage.session (cleared on browser close).
 *  - Service worker stays alive via chrome.alarms keep-alive pattern.
 */

// ─── Configuration ────────────────────────────────────────────────────────────
const CONFIG = {
  BACKEND_URL: "http://localhost:8000",
  TIER1_TIMEOUT_MS: 3000,
  TIER2_TIMEOUT_MS: 8000,
  CACHE_TTL_SECONDS: 300,         // 5-minute result cache per URL
  PHISHING_THRESHOLD: 0.65,       // Combined score above this = phishing
  SUSPICIOUS_THRESHOLD: 0.40,     // Combined score in [0.40, 0.65) = suspicious
  DEBOUNCE_MS: 800,               // Ignore rapid tab updates within this window
};

// ─── In-memory debounce map (tabId → timeout handle) ─────────────────────────
const pendingTabs = new Map();

// ─── Keep the service worker alive via a recurring alarm ─────────────────────
chrome.alarms.create("keepAlive", { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "keepAlive") {
    // No-op: just prevents SW termination during active browsing sessions
  }
});

// ─── Main tab listener ────────────────────────────────────────────────────────
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Only act when the page has fully loaded and has a real URL
  if (changeInfo.status !== "complete") return;
  if (!tab.url || tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://")) return;
  if (tab.url === "about:blank" || tab.url === "about:newtab") return;

  // Debounce: cancel any pending analysis for this tab
  if (pendingTabs.has(tabId)) {
    clearTimeout(pendingTabs.get(tabId));
  }

  const handle = setTimeout(() => {
    pendingTabs.delete(tabId);
    analyzeTab(tabId, tab.url);
  }, CONFIG.DEBOUNCE_MS);

  pendingTabs.set(tabId, handle);
});

// Clean up state when tabs are closed
chrome.tabs.onRemoved.addListener((tabId) => {
  if (pendingTabs.has(tabId)) {
    clearTimeout(pendingTabs.get(tabId));
    pendingTabs.delete(tabId);
  }
});

// ─── Core analysis orchestrator ───────────────────────────────────────────────
async function analyzeTab(tabId, url) {
  // 1. Check session cache first (avoid redundant backend calls)
  const cached = await getCachedResult(url);
  if (cached) {
    applyResult(tabId, url, cached);
    return;
  }

  // 2. TIER 1: Immediate URL analysis (target < 100ms perceived latency)
  setIcon(tabId, "scanning");

  let tier1Result = null;
  try {
    tier1Result = await withTimeout(
      fetchTier1(url),
      CONFIG.TIER1_TIMEOUT_MS
    );
  } catch (err) {
    console.warn("[PhishGuard] Tier 1 timeout or error:", err.message);
  }

  // If Tier 1 is already high-confidence phishing, warn immediately
  if (tier1Result && tier1Result.score >= CONFIG.PHISHING_THRESHOLD) {
    const result = buildResult("phishing", tier1Result.score, tier1Result, null, null);
    await cacheResult(url, result);
    applyResult(tabId, url, result);
    return; // Don't bother with DOM analysis — act fast
  }

  // 3. TIER 2: DOM/HTML contextual analysis (runs in parallel with Tier 3 kick-off)
  let domFeatures = null;
  try {
    domFeatures = await withTimeout(
      requestDOMFeatures(tabId),
      CONFIG.TIER2_TIMEOUT_MS
    );
  } catch (err) {
    console.warn("[PhishGuard] Tier 2 DOM extraction failed:", err.message);
  }

  let tier2Result = null;
  if (domFeatures) {
    try {
      tier2Result = await withTimeout(
        fetchTier2(url, domFeatures),
        CONFIG.TIER2_TIMEOUT_MS
      );
    } catch (err) {
      console.warn("[PhishGuard] Tier 2 backend call failed:", err.message);
    }
  }

  // 4. Compute fused score from Tier 1 + Tier 2
  const fusedScore = fuseScores(tier1Result, tier2Result);
  const verdict = scoreToVerdict(fusedScore);

  // Apply visual result immediately (Tier 3 is non-blocking)
  const result = buildResult(verdict, fusedScore, tier1Result, tier2Result, null);
  await cacheResult(url, result);
  applyResult(tabId, url, result);

  // 5. TIER 3: Async threat-intel enrichment (fire and forget, updates UI if positive)
  fetchTier3(url)
    .then(async (intelResult) => {
      if (!intelResult) return;
      // If threat intel confirms phishing, upgrade the verdict regardless
      const upgradedScore = Math.max(fusedScore, intelResult.in_blocklist ? 0.99 : fusedScore);
      const upgradedVerdict = scoreToVerdict(upgradedScore);
      const upgradedResult = buildResult(upgradedVerdict, upgradedScore, tier1Result, tier2Result, intelResult);
      await cacheResult(url, upgradedResult);
      // Only re-notify the tab if the verdict changed for the worse
      if (upgradedVerdict !== verdict) {
        applyResult(tabId, url, upgradedResult);
      }
    })
    .catch((err) => console.warn("[PhishGuard] Tier 3 intel failed:", err.message));
}

// ─── Tier 1: URL string analysis ─────────────────────────────────────────────
async function fetchTier1(url) {
  const resp = await fetch(`${CONFIG.BACKEND_URL}/predict/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!resp.ok) throw new Error(`Tier1 HTTP ${resp.status}`);
  return resp.json();
}

// ─── Tier 2: DOM feature submission ──────────────────────────────────────────
async function fetchTier2(url, domFeatures) {
  const resp = await fetch(`${CONFIG.BACKEND_URL}/predict/full`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, dom_features: domFeatures }),
  });
  if (!resp.ok) throw new Error(`Tier2 HTTP ${resp.status}`);
  return resp.json();
}

// ─── Tier 3: Threat intelligence lookup ──────────────────────────────────────
async function fetchTier3(url) {
  const resp = await fetch(`${CONFIG.BACKEND_URL}/predict/intel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!resp.ok) return null;
  return resp.json();
}

// ─── Request DOM features from the content script ────────────────────────────
function requestDOMFeatures(tabId) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("DOM extraction timeout")), 5000);
    chrome.tabs.sendMessage(tabId, { type: "EXTRACT_DOM_FEATURES" }, (response) => {
      clearTimeout(timeout);
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else if (response && response.features) {
        resolve(response.features);
      } else {
        reject(new Error("No DOM features returned"));
      }
    });
  });
}

// ─── Score fusion (weighted average of available tiers) ───────────────────────
function fuseScores(tier1, tier2) {
  // Weights: Tier 2 (DOM) is more reliable than pure URL analysis
  const w1 = 0.35;
  const w2 = 0.65;

  if (tier1 && tier2) {
    return w1 * tier1.score + w2 * tier2.score;
  } else if (tier2) {
    return tier2.score;
  } else if (tier1) {
    return tier1.score;
  }
  return 0.0; // No data → assume safe (minimize false positives)
}

function scoreToVerdict(score) {
  if (score >= CONFIG.PHISHING_THRESHOLD) return "phishing";
  if (score >= CONFIG.SUSPICIOUS_THRESHOLD) return "suspicious";
  return "safe";
}

function buildResult(verdict, score, tier1, tier2, intel) {
  return {
    verdict,
    score: Math.round(score * 1000) / 1000,
    tier1_score: tier1?.score ?? null,
    tier2_score: tier2?.score ?? null,
    intel_blocklisted: intel?.in_blocklist ?? false,
    intel_sources: intel?.sources ?? [],
    top_features: tier2?.top_features ?? tier1?.top_features ?? [],
    timestamp: Date.now(),
  };
}

// ─── Apply result to browser UI ───────────────────────────────────────────────
async function applyResult(tabId, url, result) {
  // Update the action icon
  setIcon(tabId, result.verdict);

  // Store result for popup retrieval
  await chrome.storage.session.set({ [`tab_${tabId}`]: result });

  // Notify content script for overlay injection (only for phishing)
  if (result.verdict === "phishing" || result.verdict === "suspicious") {
    chrome.tabs.sendMessage(tabId, {
      type: "SHOW_WARNING",
      payload: { verdict: result.verdict, score: result.score, url },
    }).catch(() => {
      // Content script may not be injected yet on some pages; ignore
    });
  }
}

// ─── Icon helpers ─────────────────────────────────────────────────────────────
const ICON_PATHS = {
  safe:      { 16: "icons/safe-16.png",      48: "icons/safe-48.png",      128: "icons/safe-128.png" },
  suspicious:{ 16: "icons/suspicious-16.png", 48: "icons/suspicious-48.png", 128: "icons/suspicious-128.png" },
  phishing:  { 16: "icons/warning-16.png",    48: "icons/warning-48.png",    128: "icons/warning-128.png" },
  scanning:  { 16: "icons/scanning-16.png",   48: "icons/scanning-48.png",   128: "icons/scanning-128.png" },
};

function setIcon(tabId, state) {
  chrome.action.setIcon({ tabId, path: ICON_PATHS[state] ?? ICON_PATHS.safe })
    .catch(() => {}); // Tab may have closed between scheduling and execution
}

// ─── Session cache helpers ────────────────────────────────────────────────────
async function getCachedResult(url) {
  const key = `cache_${btoa(url).slice(0, 64)}`;
  const data = await chrome.storage.session.get(key);
  const cached = data[key];
  if (!cached) return null;
  if (Date.now() - cached.timestamp > CONFIG.CACHE_TTL_SECONDS * 1000) return null;
  return cached;
}

async function cacheResult(url, result) {
  const key = `cache_${btoa(url).slice(0, 64)}`;
  await chrome.storage.session.set({ [key]: result });
}

// ─── Utility: timeout wrapper ─────────────────────────────────────────────────
function withTimeout(promise, ms) {
  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Operation timed out after ${ms}ms`)), ms)
  );
  return Promise.race([promise, timeout]);
}

// ─── Message handler (from popup) ────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "GET_RESULT") {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (!tabs[0]) { sendResponse(null); return; }
      const data = await chrome.storage.session.get(`tab_${tabs[0].id}`);
      sendResponse(data[`tab_${tabs[0].id}`] ?? null);
    });
    return true; // Keep the channel open for async sendResponse
  }

  if (message.type === "REANALYZE") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) analyzeTab(tabs[0].id, tabs[0].url);
    });
    sendResponse({ ok: true });
  }
});
