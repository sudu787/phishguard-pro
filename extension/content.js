/**
 * PhishGuard Pro — content.js
 *
 * Runs at document_idle (after DOM is parsed, before all resources load).
 * Responsibilities:
 *  1. Extract DOM/HTML features and return them to background.js on request.
 *  2. Render the phishing/suspicious warning overlay when instructed.
 *
 * Performance contract:
 *  - All DOM queries are batched in a single synchronous pass.
 *  - Uses requestIdleCallback for non-critical checks to avoid jank.
 *  - Never modifies the DOM unless a WARNING message is received.
 */

"use strict";

// ─── DOM Feature Extractor ────────────────────────────────────────────────────
/**
 * Extracts a rich set of HTML/DOM features for Tier 2 analysis.
 * Returns a flat object of numeric/boolean features.
 * Target execution time: < 30ms on typical pages.
 */
function extractDOMFeatures() {
  const features = {};

  // ── Forms analysis ────────────────────────────────────────────────────────
  const forms = document.querySelectorAll("form");
  features.num_forms = forms.length;
  features.has_external_form_action = 0;
  features.has_empty_form_action = 0;
  features.has_javascript_form_action = 0;
  features.num_password_inputs = document.querySelectorAll('input[type="password"]').length;
  features.num_hidden_inputs = document.querySelectorAll('input[type="hidden"]').length;
  features.has_suspicious_input_names = 0;

  const suspiciousInputNames = /passwd|password|pass|pwd|login|user|username|email|credit|card|ccnum|cvv|ssn|social/i;
  const allInputs = document.querySelectorAll("input");
  let suspiciousInputCount = 0;

  allInputs.forEach((input) => {
    const name = (input.name || input.id || "").toLowerCase();
    if (suspiciousInputNames.test(name)) suspiciousInputCount++;
  });
  features.num_suspicious_inputs = suspiciousInputCount;

  const currentHost = location.hostname;
  forms.forEach((form) => {
    const action = (form.action || "").trim().toLowerCase();
    if (!action || action === "#" || action === "javascript:void(0)") {
      features.has_empty_form_action = 1;
    } else if (action.startsWith("javascript:")) {
      features.has_javascript_form_action = 1;
    } else {
      try {
        const actionHost = new URL(form.action, location.href).hostname;
        if (actionHost && actionHost !== currentHost) {
          features.has_external_form_action = 1;
        }
      } catch (_) {
        features.has_external_form_action = 0;
      }
    }
  });

  // ── Links analysis ────────────────────────────────────────────────────────
  const links = document.querySelectorAll("a[href]");
  features.num_links = links.length;
  features.num_external_links = 0;
  features.num_null_links = 0;    // href="#" or "javascript:void(0)"
  features.pct_external_links = 0;

  // Link text vs href mismatch (lookalike phishing indicator)
  features.num_deceptive_links = 0;

  const urlPattern = /https?:\/\/[^\s]+/gi;

  links.forEach((a) => {
    const href = (a.getAttribute("href") || "").trim();
    if (!href || href === "#" || href.toLowerCase() === "javascript:void(0)") {
      features.num_null_links++;
      return;
    }
    try {
      const linkHost = new URL(href, location.href).hostname;
      if (linkHost && linkHost !== currentHost) features.num_external_links++;

      // Check if the visible text contains a URL that doesn't match the actual href
      const visibleText = a.innerText || a.textContent || "";
      const textUrls = visibleText.match(urlPattern);
      if (textUrls) {
        textUrls.forEach((textUrl) => {
          try {
            const textHost = new URL(textUrl).hostname;
            if (textHost && textHost !== linkHost) features.num_deceptive_links++;
          } catch (_) {}
        });
      }
    } catch (_) {}
  });

  if (features.num_links > 0) {
    features.pct_external_links = Math.round(
      (features.num_external_links / features.num_links) * 100
    );
  }

  // ── Script analysis ───────────────────────────────────────────────────────
  const scripts = document.querySelectorAll("script");
  features.num_scripts = scripts.length;
  features.num_external_scripts = 0;
  features.has_obfuscated_scripts = 0;

  const obfuscationPatterns = [
    /eval\s*\(/,
    /unescape\s*\(/,
    /String\.fromCharCode/,
    /\\x[0-9a-fA-F]{2}/,
    /atob\s*\(/,
    /fromCharCode/,
    /document\.write\s*\(/,
  ];

  scripts.forEach((script) => {
    if (script.src) {
      try {
        const scriptHost = new URL(script.src, location.href).hostname;
        if (scriptHost && scriptHost !== currentHost) features.num_external_scripts++;
      } catch (_) {}
    } else {
      const content = script.textContent || "";
      if (obfuscationPatterns.some((p) => p.test(content))) {
        features.has_obfuscated_scripts = 1;
      }
    }
  });

  // ── iFrame analysis ───────────────────────────────────────────────────────
  const iframes = document.querySelectorAll("iframe");
  features.num_iframes = iframes.length;
  features.num_invisible_iframes = 0;

  iframes.forEach((frame) => {
    const style = window.getComputedStyle(frame);
    const w = parseInt(frame.width || style.width) || 0;
    const h = parseInt(frame.height || style.height) || 0;
    if (w === 0 || h === 0 || style.display === "none" || style.visibility === "hidden") {
      features.num_invisible_iframes++;
    }
  });

  // ── Favicon analysis ──────────────────────────────────────────────────────
  features.favicon_from_external_domain = 0;
  const faviconLinks = document.querySelectorAll('link[rel*="icon"]');
  faviconLinks.forEach((link) => {
    try {
      const faviconHost = new URL(link.href, location.href).hostname;
      if (faviconHost && faviconHost !== currentHost) {
        features.favicon_from_external_domain = 1;
      }
    } catch (_) {}
  });

  // ── Meta / page characteristics ───────────────────────────────────────────
  const metaRefresh = document.querySelector('meta[http-equiv="refresh"]');
  features.has_meta_refresh = metaRefresh ? 1 : 0;

  // Right-click / copy-paste disabling
  features.disables_right_click = document.body
    ? /contextmenu.*return\s*false/i.test(document.body.getAttribute("oncontextmenu") || "")
    : 0;

  // Page has a visible title?
  features.has_empty_title = (!document.title || document.title.trim().length === 0) ? 1 : 0;

  // Copyright text typically indicates legitimate site
  const bodyText = (document.body?.innerText || "").toLowerCase();
  features.has_copyright = /©|copyright|\bcopr\b/i.test(bodyText) ? 1 : 0;

  // Login-specific keywords
  features.has_login_keywords = /sign\s*in|log\s*in|login|verify|confirm|secure|account|credential/i.test(bodyText) ? 1 : 0;

  // ── Redirect / popup behavior ─────────────────────────────────────────────
  features.has_status_bar_customization = 0;
  const inlineHandlers = document.querySelectorAll("[onmouseover]");
  inlineHandlers.forEach((el) => {
    if (/window\.status/i.test(el.getAttribute("onmouseover") || "")) {
      features.has_status_bar_customization = 1;
    }
  });

  // ── Page metadata ─────────────────────────────────────────────────────────
  features.page_size_bytes = (document.documentElement?.innerHTML || "").length;
  features.dom_depth = getDOMDepth(document.body, 0);

  return features;
}

/** Estimate max DOM depth (capped at 50 nodes to avoid stack overflow) */
function getDOMDepth(node, depth) {
  if (!node || depth > 50) return depth;
  let max = depth;
  for (const child of node.children || []) {
    max = Math.max(max, getDOMDepth(child, depth + 1));
  }
  return max;
}

// ─── Warning Overlay ──────────────────────────────────────────────────────────
function showWarningBanner(payload) {
  // Don't double-inject
  if (document.getElementById("phishguard-banner")) return;

  const { verdict, score, url } = payload;
  const isPhishing = verdict === "phishing";
  const bgColor = isPhishing ? "#c0392b" : "#e67e22";
  const icon = isPhishing ? "🚨" : "⚠️";
  const title = isPhishing ? "Phishing Website Detected" : "Suspicious Website";
  const pct = Math.round(score * 100);

  const banner = document.createElement("div");
  banner.id = "phishguard-banner";
  banner.setAttribute("role", "alert");
  banner.setAttribute("aria-live", "assertive");
  banner.style.cssText = `
    all: initial;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 2147483647 !important;
    background: ${bgColor} !important;
    color: #fff !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-size: 14px !important;
    padding: 12px 48px 12px 16px !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.35) !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    box-sizing: border-box !important;
  `;

  banner.innerHTML = `
    <span style="font-size:20px">${icon}</span>
    <div style="flex:1">
      <strong style="display:block;font-size:15px">${title}</strong>
      <span style="opacity:0.85">Risk score: ${pct}% — This page may be attempting to steal your credentials. Proceed with caution.</span>
    </div>
    <button id="phishguard-close"
      aria-label="Dismiss warning"
      style="all:initial;position:absolute;right:12px;top:50%;transform:translateY(-50%);
             background:rgba(255,255,255,0.25);border:none;color:#fff;font-size:20px;
             cursor:pointer;padding:4px 10px;border-radius:4px;line-height:1">×</button>
  `;

  document.documentElement.prepend(banner);

  document.getElementById("phishguard-close").addEventListener("click", (e) => {
    e.stopPropagation();
    banner.remove();
  });

  // Auto-dismiss suspicious (not phishing) banners after 8s
  if (!isPhishing) setTimeout(() => banner.remove(), 8000);
}

// ─── Message listener ─────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "EXTRACT_DOM_FEATURES") {
    try {
      const features = extractDOMFeatures();
      sendResponse({ features });
    } catch (err) {
      sendResponse({ features: null, error: err.message });
    }
    return false; // Synchronous response — no need to keep channel open
  }

  if (message.type === "SHOW_WARNING") {
    showWarningBanner(message.payload);
    sendResponse({ ok: true });
    return false;
  }
});
