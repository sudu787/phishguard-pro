/**
 * PhishGuard V2 — Content Script (DOM Feature Extractor)
 * content/content.js
 *
 * Runs at document_idle on every HTTP/HTTPS page.
 * Loaded AFTER warning-overlay.js (shares the same content script world).
 *
 * Responsibilities:
 *   1. Extract 26 DOM/HTML features on request from the service worker
 *   2. Display warning overlays when instructed by the service worker
 *
 * Performance:
 *   - All DOM queries are batched in a single synchronous pass
 *   - Target execution: < 30ms on typical pages
 *   - Never modifies the DOM unless a SHOW_WARNING is received
 *   - getDOMDepth is bounded (max 50) to prevent stack overflow
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════════════════
//  DOM Feature Extraction
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Extract a comprehensive set of DOM/HTML features for phishing detection.
 * Returns a flat object of numeric values matching the backend's DOMFeatures schema.
 *
 * @return {Object} — Feature map (all values are numbers: 0/1 for boolean, counts, percentages)
 */
function extractDOMFeatures() {
  const t0 = performance.now();
  const features = {};
  const currentHost = location.hostname;

  // ── 1. Form Analysis ────────────────────────────────────────────────────
  const forms = document.querySelectorAll('form');
  features.num_forms = forms.length;
  features.has_external_form_action = 0;
  features.has_empty_form_action = 0;
  features.has_javascript_form_action = 0;

  forms.forEach((form) => {
    const action = (form.getAttribute('action') || '').trim().toLowerCase();

    if (!action || action === '#' || action === 'javascript:void(0)' || action === 'about:blank') {
      features.has_empty_form_action = 1;
    } else if (action.startsWith('javascript:')) {
      features.has_javascript_form_action = 1;
    } else {
      try {
        const actionUrl = new URL(form.action, location.href);
        if (actionUrl.hostname && actionUrl.hostname !== currentHost) {
          features.has_external_form_action = 1;
        }
      } catch {
        // Malformed action URL — treat as suspicious
        features.has_empty_form_action = 1;
      }
    }
  });

  // ── 2. Input Field Analysis ─────────────────────────────────────────────
  features.num_password_inputs = document.querySelectorAll('input[type="password"]').length;
  features.num_hidden_inputs   = document.querySelectorAll('input[type="hidden"]').length;

  const suspiciousInputPattern = /passwd|password|pass|pwd|login|user|username|email|credit|card|ccnum|cvv|ssn|social|pin|otp|mfa/i;
  const allInputs = document.querySelectorAll('input');
  let suspiciousInputCount = 0;

  allInputs.forEach((input) => {
    const identifier = (input.name || input.id || input.placeholder || '').toLowerCase();
    if (suspiciousInputPattern.test(identifier)) suspiciousInputCount++;
  });
  features.num_suspicious_inputs = suspiciousInputCount;

  // ── 3. Link Analysis ────────────────────────────────────────────────────
  const links = document.querySelectorAll('a[href]');
  features.num_links = links.length;
  features.num_external_links = 0;
  features.num_null_links = 0;
  features.pct_external_links = 0;
  features.num_deceptive_links = 0;

  const urlPattern = /https?:\/\/[^\s"'<>]+/gi;

  links.forEach((a) => {
    const href = (a.getAttribute('href') || '').trim();

    // Null/empty links (non-functional — common in phishing clones)
    if (!href || href === '#' || href === 'javascript:void(0)' || href === 'javascript:;') {
      features.num_null_links++;
      return;
    }

    try {
      const linkUrl = new URL(href, location.href);
      if (linkUrl.hostname && linkUrl.hostname !== currentHost) {
        features.num_external_links++;
      }

      // Deceptive links: visible text shows a URL different from actual href
      const visibleText = (a.textContent || '').trim();
      const textUrls = visibleText.match(urlPattern);
      if (textUrls) {
        for (const textUrl of textUrls) {
          try {
            const textHost = new URL(textUrl).hostname;
            if (textHost && textHost !== linkUrl.hostname) {
              features.num_deceptive_links++;
              break; // Count max 1 per link element
            }
          } catch { /* ignore parse failures */ }
        }
      }
    } catch { /* ignore malformed hrefs */ }
  });

  if (features.num_links > 0) {
    features.pct_external_links = Math.round(
      (features.num_external_links / features.num_links) * 100
    );
  }

  // ── 4. Script Analysis ──────────────────────────────────────────────────
  const scripts = document.querySelectorAll('script');
  features.num_scripts = scripts.length;
  features.num_external_scripts = 0;
  features.has_obfuscated_scripts = 0;

  const obfuscationPatterns = [
    /\beval\s*\(/,
    /\bunescape\s*\(/,
    /String\.fromCharCode/,
    /\\x[0-9a-fA-F]{2}/,
    /\batob\s*\(/,
    /\bbtoa\s*\(/,
    /document\.write\s*\(/,
    /\\u[0-9a-fA-F]{4}/,
    /Function\s*\(\s*['"`]/,
  ];

  scripts.forEach((script) => {
    if (script.src) {
      try {
        const scriptUrl = new URL(script.src, location.href);
        if (scriptUrl.hostname && scriptUrl.hostname !== currentHost) {
          features.num_external_scripts++;
        }
      } catch { /* ignore */ }
    } else {
      const content = script.textContent || '';
      if (content.length > 0 && obfuscationPatterns.some(p => p.test(content))) {
        features.has_obfuscated_scripts = 1;
      }
    }
  });

  // ── 5. iframe Analysis ──────────────────────────────────────────────────
  const iframes = document.querySelectorAll('iframe');
  features.num_iframes = iframes.length;
  features.num_invisible_iframes = 0;

  iframes.forEach((frame) => {
    try {
      const style = window.getComputedStyle(frame);
      const w = parseInt(frame.width || style.width) || 0;
      const h = parseInt(frame.height || style.height) || 0;
      const isHidden = (
        w <= 1 || h <= 1 ||
        style.display === 'none' ||
        style.visibility === 'hidden' ||
        parseFloat(style.opacity) === 0
      );
      if (isHidden) features.num_invisible_iframes++;
    } catch {
      // Cross-origin style access may fail — treat as suspicious
      features.num_invisible_iframes++;
    }
  });

  // ── 6. Favicon Analysis ─────────────────────────────────────────────────
  features.favicon_from_external_domain = 0;
  const faviconLinks = document.querySelectorAll('link[rel*="icon"]');
  faviconLinks.forEach((link) => {
    try {
      const faviconUrl = new URL(link.href, location.href);
      if (faviconUrl.hostname && faviconUrl.hostname !== currentHost) {
        features.favicon_from_external_domain = 1;
      }
    } catch { /* ignore */ }
  });

  // ── 7. Meta / Page Characteristics ──────────────────────────────────────
  features.has_meta_refresh = document.querySelector('meta[http-equiv="refresh"]') ? 1 : 0;

  // Right-click disabled (hiding page source/inspection)
  const bodyContextMenu = document.body?.getAttribute('oncontextmenu') || '';
  features.disables_right_click = /return\s*false/i.test(bodyContextMenu) ? 1 : 0;

  // Empty title (clone pages often forget to set a title)
  features.has_empty_title = (!document.title || document.title.trim().length === 0) ? 1 : 0;

  // Copyright notice (legitimate sites almost always have one)
  const bodyText = (document.body?.innerText || '').substring(0, 10000).toLowerCase();
  features.has_copyright = /©|copyright|\bcopr\b|\(c\)\s*\d{4}/i.test(bodyText) ? 1 : 0;

  // Login-specific keywords in page body
  features.has_login_keywords = /sign\s*in|log\s*in|login|verify your|confirm your|secure your|your account|your credential/i.test(bodyText) ? 1 : 0;

  // ── 8. Status Bar Manipulation ──────────────────────────────────────────
  features.has_status_bar_customization = 0;
  const onMouseOverEls = document.querySelectorAll('[onmouseover]');
  onMouseOverEls.forEach((el) => {
    if (/window\.status/i.test(el.getAttribute('onmouseover') || '')) {
      features.has_status_bar_customization = 1;
    }
  });

  // ── 9. Page Metadata ────────────────────────────────────────────────────
  features.page_size_bytes = (document.documentElement?.innerHTML || '').length;
  features.dom_depth = _getDOMDepth(document.body, 0);

  // ── 10. Page text for the Tier 2 page model (same "title | text" form as fetch_phreshphish_html.py)
  const visible = (document.body?.innerText || '').replace(/\s+/g, ' ').trim();
  features.page_text = `${(document.title || '').trim()} | ${visible}`.substring(0, 2000);

  // ── Performance logging ─────────────────────────────────────────────────
  const elapsed = Math.round(performance.now() - t0);
  if (elapsed > 50) {
    console.warn(`[PhishGuard] DOM extraction took ${elapsed}ms (target: <30ms)`);
  }

  return features;
}


/**
 * Estimate maximum DOM tree depth.
 * Capped at 50 to prevent stack overflow on deeply nested pages.
 *
 * @param  {Node}   node  — Starting node
 * @param  {number} depth — Current depth
 * @return {number} — Maximum depth found
 */
function _getDOMDepth(node, depth) {
  if (!node || depth > 50) return depth;
  let maxDepth = depth;
  const children = node.children;
  if (children) {
    for (let i = 0; i < children.length && i < 100; i++) {
      const childDepth = _getDOMDepth(children[i], depth + 1);
      if (childDepth > maxDepth) maxDepth = childDepth;
    }
  }
  return maxDepth;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Message Handler
// ═══════════════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {

  // ── Service worker requests DOM features ────────────────────────────────
  if (message.type === 'EXTRACT_DOM_FEATURES') {
    try {
      const features = extractDOMFeatures();
      
      // Run deep heuristic analysis
      let localHeuristics = null;
      if (window.PageAnalyzer) {
         const analyzer = new window.PageAnalyzer();
         localHeuristics = analyzer.analyze();
         
         // Trigger immediate local warning if critical exfiltration detected
         if (localHeuristics.criticalThreat) {
            console.warn('[PhishGuard] Critical local threat detected! Triggering immediate warning.');
            if (typeof showWarningOverlay === 'function') {
               showWarningOverlay({
                  verdict: 'phishing',
                  score: 0.99,
                  features: localHeuristics.signals.map(s => ({
                     name: s.name,
                     label: `Local Heuristic: ${s.name}`,
                     severity: s.severity,
                     weight: 1.0
                  }))
               });
            }
         }
      }

      sendResponse({ features, localHeuristics });
    } catch (err) {
      console.error('[PhishGuard] DOM extraction error:', err);
      sendResponse({ features: null, error: err.message });
    }
    return false; // Synchronous response
  }

  // ── Service worker requests warning display ─────────────────────────────
  if (message.type === 'SHOW_WARNING') {
    try {
      // showWarningOverlay is defined in warning-overlay.js (loaded before this file)
      showWarningOverlay(message.payload);
      sendResponse({ ok: true });
    } catch (err) {
      console.error('[PhishGuard] Warning overlay error:', err);
      sendResponse({ ok: false, error: err.message });
    }
    return false;
  }

  // ── Remove warning (e.g., after re-scan shows safe) ────────────────────
  if (message.type === 'REMOVE_WARNING') {
    removeWarningOverlay();
    sendResponse({ ok: true });
    return false;
  }
});
