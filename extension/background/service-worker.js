/**
 * PhishGuard V2 — Background Service Worker
 * background/service-worker.js
 *
 * Main orchestrator for the phishing detection pipeline.
 *
 * Flow:
 *   1. Tab navigation detected → debounce rapid updates
 *   2. Check cache → return immediately if fresh result exists
 *   3. Phase A: Local URL analysis (instant, ~1ms)
 *   4. If URL score ≥ PHISHING_THRESHOLD → warn immediately (fast path)
 *   5. Phase B: Request DOM features from content script (~10–30ms)
 *   6. Fuse URL + DOM scores → final verdict
 *   7. Apply result: badge icon + content script warning overlay
 *   8. Cache result for popup and future navigations
 *
 * Design principles:
 *   - Never blocks the browser UI thread — all work is async
 *   - Debouncing prevents duplicate scans on rapid navigations
 *   - Graceful degradation: if DOM extraction fails, URL score alone is used
 *   - Service worker keep-alive via chrome.alarms
 */

import { URLAnalyzer } from './url-analyzer.js';
import { CacheManager } from './cache.js';

// ─── Instantiate modules ─────────────────────────────────────────────────────
const analyzer = new URLAnalyzer();
const cache    = new CacheManager(5 * 60 * 1000); // 5-minute TTL

// ─── Configuration ────────────────────────────────────────────────────────────
const CONFIG = {
  PHISHING_THRESHOLD:   0.65,
  SUSPICIOUS_THRESHOLD: 0.40,
  DEBOUNCE_MS:          600,
  DOM_TIMEOUT_MS:       5000,
  URL_WEIGHT:           0.40,   // URL score contributes 40% to fused score
  DOM_WEIGHT:           0.60,   // DOM score contributes 60% to fused score
};

// ─── Internal state ───────────────────────────────────────────────────────────
const pendingTabs = new Map();  // tabId → debounce timeout handle

// ─── Keep service worker alive ────────────────────────────────────────────────
chrome.alarms.create('keepAlive', { periodInMinutes: 0.4 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'keepAlive') {
    // No-op — prevents MV3 service worker from being terminated during active use
  }
});


// ═══════════════════════════════════════════════════════════════════════════════
//  Tab Navigation Listener
// ═══════════════════════════════════════════════════════════════════════════════

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Only trigger when the page has fully loaded
  if (changeInfo.status !== 'complete') return;

  // Skip internal Chrome pages and blank tabs
  if (!tab.url || isInternalUrl(tab.url)) return;

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

// Clean up debounce state when tabs close
chrome.tabs.onRemoved.addListener((tabId) => {
  if (pendingTabs.has(tabId)) {
    clearTimeout(pendingTabs.get(tabId));
    pendingTabs.delete(tabId);
  }
});


// ═══════════════════════════════════════════════════════════════════════════════
//  Core Analysis Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

async function analyzeTab(tabId, url) {
  const t0 = performance.now();

  // ── Step 1: Check cache ─────────────────────────────────────────────────
  const cached = await cache.get(url);
  if (cached) {
    await applyResult(tabId, cached);
    return;
  }

  // ── Step 2: Set scanning state ──────────────────────────────────────────
  setBadge(tabId, 'scanning');

  // ── Step 3: Request DOM features from content script ────────────────────
  let domFeatures = null;
  try {
    domFeatures = await withTimeout(
      requestDOMFeatures(tabId),
      CONFIG.DOM_TIMEOUT_MS
    );
  } catch (err) {
    console.warn('[PhishGuard] DOM extraction failed:', err.message);
    // Continue without DOM — graceful degradation
  }

  // ── Step 4: Call Backend ML API (New integration!) ──────────────────────
  let result = null;
  try {
    const response = await fetch('http://127.0.0.1:8000/predict/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: url,
        dom_features: domFeatures || {}
      })
    });
    
    if (response.ok) {
      const backendData = await response.json();
      // Translate backend response to extension format
      result = {
        url,
        verdict: backendData.risk_score >= 65 ? 'phishing' : backendData.risk_score >= 40 ? 'suspicious' : 'safe',
        score: backendData.risk_score / 100, // Extension expects 0-1
        urlScore: backendData.risk_score / 100, 
        domScore: backendData.risk_score / 100,
        features: backendData.reasons.map(reason => ({
          name: 'AI Agent Finding',
          label: reason,
          severity: 'high',
          weight: 1.0
        })),
        socWarning: backendData.user_warning, // Store the AI generated warning
        timestamp: Date.now(),
        scanDuration: Math.round(performance.now() - t0),
        isAiReport: true
      };
      console.log(`[PhishGuard] BACKEND SUCCESS: ${url.slice(0, 60)} → ${result.verdict} (${result.score})`);
    } else {
      console.warn(`[PhishGuard] Backend returned ${response.status}. Falling back to local heuristic.`);
    }
  } catch (err) {
    console.warn(`[PhishGuard] Backend unreachable (${err.message}). Falling back to local heuristic.`);
  }

  // ── Step 5: Graceful Fallback to Local Heuristics ───────────────────────
  if (!result) {
    const urlResult = analyzer.analyze(url);
    result = buildResult(url, urlResult, domFeatures, t0);
  }

  // ── Step 6: Cache and Apply ─────────────────────────────────────────────
  await cache.set(url, result);
  await cache.setTabResult(tabId, result);
  await applyResult(tabId, result);

  console.log(`[PhishGuard] ${url.slice(0, 60)} → ${result.verdict} (${result.score}) in ${result.scanDuration}ms`);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Score Fusion & Result Building
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Build the final ScanResult by fusing URL and DOM analysis.
 *
 * Score fusion strategy:
 *   - If both URL and DOM scores are available:
 *       fused = URL_WEIGHT × urlScore + DOM_WEIGHT × domScore
 *   - If only URL score: fused = urlScore (no penalty for missing DOM)
 *   - DOM features can push an ambiguous URL score into phishing territory
 */
function buildResult(url, urlResult, domFeatures, t0) {
  let fusedScore = urlResult.score;
  let domScore = null;
  let allFeatures = [...urlResult.features];

  if (domFeatures) {
    domScore = computeDOMRiskScore(domFeatures);
    fusedScore = CONFIG.URL_WEIGHT * urlResult.score + CONFIG.DOM_WEIGHT * domScore;

    // Add DOM features to the feature list
    const domFeaturesList = buildDOMFeatureList(domFeatures);
    allFeatures = allFeatures.concat(domFeaturesList);
  }

  // Clamp score to [0, 1]
  fusedScore = Math.max(0, Math.min(1, fusedScore));
  fusedScore = Math.round(fusedScore * 1000) / 1000;

  const verdict = scoreToVerdict(fusedScore);

  // Sort all features by severity/weight for display
  allFeatures.sort((a, b) => b.weight - a.weight);

  return {
    url,
    verdict,
    score: fusedScore,
    urlScore: urlResult.score,
    domScore,
    features: allFeatures.slice(0, 12), // Top 12 features for popup
    timestamp: Date.now(),
    scanDuration: Math.round(performance.now() - t0),
  };
}


/**
 * Compute a risk score from DOM features.
 * Weights are assigned to each DOM signal based on phishing correlation.
 */
function computeDOMRiskScore(dom) {
  let rawScore = 0;

  // External form action — strongest DOM signal
  if (dom.has_external_form_action)     rawScore += 3.0;
  if (dom.has_empty_form_action)        rawScore += 1.5;
  if (dom.has_javascript_form_action)   rawScore += 2.0;

  // Password inputs without HTTPS
  if (dom.num_password_inputs > 0)      rawScore += 1.0;
  if (dom.num_password_inputs > 2)      rawScore += 1.5;

  // Suspicious inputs (named "password", "ssn", etc.)
  if (dom.num_suspicious_inputs > 0)    rawScore += 1.0;

  // Hidden inputs (often used to pass tokens/session IDs to phishing servers)
  if (dom.num_hidden_inputs > 5)        rawScore += 1.0;

  // Deceptive links (visible text ≠ actual href)
  if (dom.num_deceptive_links > 0)      rawScore += 2.5;

  // External link ratio
  if (dom.pct_external_links > 80)      rawScore += 1.5;
  else if (dom.pct_external_links > 50) rawScore += 0.8;

  // Obfuscated scripts (eval, fromCharCode, etc.)
  if (dom.has_obfuscated_scripts)       rawScore += 0.5;

  // Invisible iframes
  if (dom.num_invisible_iframes > 0)    rawScore += 2.0;

  // External favicon (impersonation signal)
  if (dom.favicon_from_external_domain) rawScore += 1.5;

  // Meta refresh redirect
  if (dom.has_meta_refresh)             rawScore += 1.5;

  // Right-click disabled (hiding context)
  if (dom.disables_right_click)         rawScore += 1.5;

  // Empty title
  if (dom.has_empty_title)              rawScore += 0.8;

  // Login keywords without copyright (fake page signal)
  if (dom.has_login_keywords && !dom.has_copyright) rawScore += 1.5;

  // Status bar customization
  if (dom.has_status_bar_customization) rawScore += 1.5;

  // Null/empty links (non-functional page)
  const nullLinkRatio = dom.num_links > 0 ? dom.num_null_links / dom.num_links : 0;
  if (nullLinkRatio > 0.5)             rawScore += 1.5;

  // Sigmoid normalisation (same curve as URL analyzer)
  return 1 / (1 + Math.exp(-1.0 * (rawScore - 3.5)));
}


/** Convert DOM analysis features into displayable feature objects. */
function buildDOMFeatureList(dom) {
  const features = [];

  if (dom.has_external_form_action) {
    features.push({ name: 'external_form', label: 'Login form submits data to an external server', severity: 'critical', weight: 3.0 });
  }
  if (dom.has_empty_form_action) {
    features.push({ name: 'empty_form_action', label: 'Form has no action URL — data may be captured by scripts', severity: 'medium', weight: 1.5 });
  }
  if (dom.has_javascript_form_action) {
    features.push({ name: 'js_form_action', label: 'Form action uses JavaScript — possible data interception', severity: 'high', weight: 2.0 });
  }
  if (dom.num_deceptive_links > 0) {
    features.push({ name: 'deceptive_links', label: `${dom.num_deceptive_links} link(s) show a different URL than where they lead`, severity: 'high', weight: 2.5 });
  }
  if (dom.has_obfuscated_scripts) {
    features.push({ name: 'obfuscated_scripts', label: 'Page contains obfuscated JavaScript (eval, fromCharCode)', severity: 'high', weight: 2.5 });
  }
  if (dom.num_invisible_iframes > 0) {
    features.push({ name: 'hidden_iframes', label: `${dom.num_invisible_iframes} hidden iframe(s) detected — may load malicious content`, severity: 'high', weight: 2.0 });
  }
  if (dom.favicon_from_external_domain) {
    features.push({ name: 'external_favicon', label: 'Favicon loaded from a different domain — possible brand impersonation', severity: 'medium', weight: 1.5 });
  }
  if (dom.has_meta_refresh) {
    features.push({ name: 'meta_refresh', label: 'Page has auto-redirect via meta refresh', severity: 'medium', weight: 1.5 });
  }
  if (dom.disables_right_click) {
    features.push({ name: 'no_right_click', label: 'Right-click is disabled — hiding page inspection', severity: 'medium', weight: 1.5 });
  }
  if (dom.has_login_keywords && !dom.has_copyright) {
    features.push({ name: 'login_no_copyright', label: 'Login page without copyright notice — likely a clone', severity: 'medium', weight: 1.5 });
  }
  if (dom.num_password_inputs > 0) {
    features.push({ name: 'password_fields', label: `${dom.num_password_inputs} password field(s) on this page`, severity: 'low', weight: 1.0 });
  }
  if (dom.has_status_bar_customization) {
    features.push({ name: 'status_bar', label: 'Status bar is being manipulated to hide link destinations', severity: 'medium', weight: 1.5 });
  }

  return features;
}


function scoreToVerdict(score) {
  if (score >= CONFIG.PHISHING_THRESHOLD) return 'phishing';
  if (score >= CONFIG.SUSPICIOUS_THRESHOLD) return 'suspicious';
  return 'safe';
}


// ═══════════════════════════════════════════════════════════════════════════════
//  DOM Feature Request (Content Script Communication)
// ═══════════════════════════════════════════════════════════════════════════════

function requestDOMFeatures(tabId) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, { type: 'EXTRACT_DOM_FEATURES' }, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (response && response.features) {
        resolve(response.features);
      } else {
        reject(new Error('No DOM features returned'));
      }
    });
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
//  UI Feedback (Badge + Warning Overlay)
// ═══════════════════════════════════════════════════════════════════════════════

async function applyResult(tabId, result) {
  // Update the action badge
  setBadge(tabId, result.verdict);

  // Store result for popup retrieval
  await cache.setTabResult(tabId, result);

  // Notify content script for warning overlay
  if (result.verdict === 'phishing' || result.verdict === 'suspicious') {
    try {
      await chrome.tabs.sendMessage(tabId, {
        type: 'SHOW_WARNING',
        payload: {
          verdict: result.verdict,
          score: result.score,
          url: result.url,
          features: result.features.slice(0, 5), // Top 5 for overlay
        },
      });
    } catch {
      // Content script may not be injected yet — non-critical
    }
  }
}


/** Set the extension badge text and color based on scan state. */
function setBadge(tabId, state) {
  const badgeConfig = {
    safe:       { text: '✓',  bg: '#10b981' },
    suspicious: { text: '!',  bg: '#f59e0b' },
    phishing:   { text: '✕',  bg: '#ef4444' },
    scanning:   { text: '…',  bg: '#6366f1' },
  };

  const config = badgeConfig[state] || badgeConfig.safe;

  chrome.action.setBadgeText({ tabId, text: config.text }).catch(() => {});
  chrome.action.setBadgeBackgroundColor({ tabId, color: config.bg }).catch(() => {});
  chrome.action.setBadgeTextColor({ tabId, color: '#ffffff' }).catch(() => {});
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Message Handler (Popup + Content Script Communication)
// ═══════════════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  // Popup requests current tab's scan result
  if (message.type === 'GET_RESULT') {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (!tabs[0]) {
        sendResponse(null);
        return;
      }
      const result = await cache.getTabResult(tabs[0].id);
      sendResponse(result);
    });
    return true; // Keep channel open for async response
  }

  // Popup requests a re-scan of the current tab
  if (message.type === 'REANALYZE') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0] && tabs[0].url) {
        // Clear cache for this URL first
        cache.get(tabs[0].url).then(() => {
          analyzeTab(tabs[0].id, tabs[0].url);
        });
      }
    });
    sendResponse({ ok: true });
    return false;
  }

  // Popup requests scan history (last N scans)
  if (message.type === 'GET_HISTORY') {
    // History is maintained via session storage; return recent tab results
    sendResponse({ history: [] }); // Placeholder — will implement in Phase 2
    return false;
  }
});


// ═══════════════════════════════════════════════════════════════════════════════
//  Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

/** Check if a URL is an internal Chrome page that should not be scanned. */
function isInternalUrl(url) {
  const internalPrefixes = [
    'chrome://',
    'chrome-extension://',
    'about:',
    'edge://',
    'brave://',
    'chrome-search://',
    'devtools://',
    'view-source:',
  ];
  return internalPrefixes.some(prefix => url.startsWith(prefix));
}

/** Promise.race with a timeout — rejects if the promise doesn't resolve in time. */
function withTimeout(promise, ms) {
  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Timed out after ${ms}ms`)), ms)
  );
  return Promise.race([promise, timeout]);
}


// ─── Startup log ──────────────────────────────────────────────────────────────
console.log('[PhishGuard V2] Service worker loaded — local URL analysis active');
