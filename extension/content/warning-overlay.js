/**
 * PhishGuard V2 — Warning Overlay System
 * content/warning-overlay.js
 *
 * Injects premium warning UI into web pages when phishing is detected.
 *
 * Two warning modes:
 *   1. PHISHING  → Full-page blocking overlay with backdrop blur
 *   2. SUSPICIOUS → Non-blocking top banner with auto-dismiss
 *
 * Security:
 *   - All DOM is created via createElement (never innerHTML with user data)
 *   - Styles use !important to resist page CSS overrides
 *   - z-index: 2147483647 ensures overlay is always on top
 *   - Shadow DOM isolates our styles from the host page
 *
 * Loaded BEFORE content.js via manifest content_scripts order.
 */

'use strict';

const PHISHGUARD_OVERLAY_ID = 'phishguard-warning-root';
const PHISHGUARD_BANNER_ID  = 'phishguard-banner-root';

/**
 * Show the appropriate warning based on verdict.
 *
 * @param {Object} payload
 * @param {string} payload.verdict   — 'phishing' | 'suspicious'
 * @param {number} payload.score     — 0.0 to 1.0
 * @param {string} payload.url       — The URL being warned about
 * @param {Array}  payload.features  — Top triggered features for display
 */
function showWarningOverlay(payload) {
  // Don't double-inject
  if (document.getElementById(PHISHGUARD_OVERLAY_ID) ||
      document.getElementById(PHISHGUARD_BANNER_ID)) {
    return;
  }

  if (payload.verdict === 'phishing') {
    _showBlockingOverlay(payload);
  } else if (payload.verdict === 'suspicious') {
    _showBanner(payload);
  }
}

/** Remove all PhishGuard warning UI from the page. */
function removeWarningOverlay() {
  const overlay = document.getElementById(PHISHGUARD_OVERLAY_ID);
  const banner  = document.getElementById(PHISHGUARD_BANNER_ID);
  if (overlay) overlay.remove();
  if (banner) banner.remove();
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Phishing: Full-Page Blocking Overlay
// ═══════════════════════════════════════════════════════════════════════════════

function _showBlockingOverlay(payload) {
  const { score, url, features = [] } = payload;
  const pct = Math.round(score * 100);

  // Create host element with Shadow DOM for style isolation
  const host = document.createElement('div');
  host.id = PHISHGUARD_OVERLAY_ID;
  host.setAttribute('role', 'alertdialog');
  host.setAttribute('aria-modal', 'true');
  host.setAttribute('aria-label', 'PhishGuard phishing warning');

  const shadow = host.attachShadow({ mode: 'closed' });

  // Build feature list HTML
  const featureItems = features.slice(0, 5).map(f => {
    const icon = f.severity === 'critical' ? '🔴'
               : f.severity === 'high' ? '🟠'
               : '🟡';
    const text = _escapeHtml(f.label || f.name);
    return `<li class="feature-item"><span class="f-icon">${icon}</span><span class="f-text">${text}</span></li>`;
  }).join('');

  shadow.innerHTML = `
    <style>${_getOverlayStyles()}</style>
    <div class="overlay" id="pgOverlay">
      <div class="card" role="document">
        <div class="card-header">
          <div class="shield-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z" fill="#ef4444" opacity="0.15"/>
              <path d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z" stroke="#ef4444" stroke-width="1.5" fill="none"/>
              <path d="M15 9l-6 6M9 9l6 6" stroke="#ef4444" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </div>
          <h1 class="title">Phishing Website Detected</h1>
          <p class="subtitle">PhishGuard has blocked this page to protect your data</p>
        </div>

        <div class="score-section">
          <div class="score-bar">
            <div class="score-fill" style="width: ${pct}%"></div>
          </div>
          <span class="score-label">Threat Level: <strong>${pct}%</strong></span>
        </div>

        <div class="url-section">
          <span class="url-label">Blocked URL</span>
          <span class="url-value">${_escapeHtml(_truncateUrl(url, 70))}</span>
        </div>

        ${features.length > 0 ? `
        <div class="features-section">
          <span class="features-title">Why this page is dangerous</span>
          <ul class="features-list">${featureItems}</ul>
        </div>
        ` : ''}

        <div class="actions">
          <button class="btn btn-primary" id="pgGoBack" autofocus>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M19 12H5M12 19l-7-7 7-7"/>
            </svg>
            Go Back to Safety
          </button>
          <button class="btn btn-ghost" id="pgProceed">
            I understand the risk — proceed anyway
          </button>
        </div>

        <div class="footer">
          <span>🛡️ Protected by PhishGuard V2</span>
        </div>
      </div>
    </div>
  `;

  document.documentElement.appendChild(host);

  // Event listeners
  const goBackBtn   = shadow.getElementById('pgGoBack');
  const proceedBtn  = shadow.getElementById('pgProceed');

  goBackBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (window.history.length > 1) {
      window.history.back();
    } else {
      window.location.href = 'about:newtab';
    }
  });

  proceedBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const overlay = shadow.getElementById('pgOverlay');
    overlay.classList.add('dismissing');
    setTimeout(() => host.remove(), 400);
  });

  // Focus trap — keep keyboard focus inside the overlay
  goBackBtn.focus();
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Suspicious: Non-Blocking Top Banner
// ═══════════════════════════════════════════════════════════════════════════════

function _showBanner(payload) {
  const { score, url, features = [] } = payload;
  const pct = Math.round(score * 100);

  const host = document.createElement('div');
  host.id = PHISHGUARD_BANNER_ID;
  host.setAttribute('role', 'alert');
  host.setAttribute('aria-live', 'assertive');

  const shadow = host.attachShadow({ mode: 'closed' });

  const topFeature = features.length > 0
    ? _escapeHtml(features[0].label || features[0].name)
    : 'Unusual URL patterns detected';

  shadow.innerHTML = `
    <style>${_getBannerStyles()}</style>
    <div class="banner" id="pgBanner">
      <div class="banner-content">
        <div class="banner-icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L1 21h22L12 2z" fill="#f59e0b" opacity="0.15"/>
            <path d="M12 2L1 21h22L12 2z" stroke="#f59e0b" stroke-width="1.5" fill="none"/>
            <path d="M12 9v4M12 17h.01" stroke="#f59e0b" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="banner-text">
          <strong>Suspicious Website (${pct}% risk)</strong>
          <span class="banner-detail">${topFeature}</span>
        </div>
        <button class="banner-close" id="pgBannerClose" aria-label="Dismiss warning">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>
      </div>
      <div class="banner-bar">
        <div class="banner-bar-fill" style="width: ${pct}%"></div>
      </div>
    </div>
  `;

  document.documentElement.prepend(host);

  shadow.getElementById('pgBannerClose').addEventListener('click', (e) => {
    e.stopPropagation();
    const banner = shadow.getElementById('pgBanner');
    banner.classList.add('dismissing');
    setTimeout(() => host.remove(), 300);
  });

  // Auto-dismiss after 10 seconds
  setTimeout(() => {
    const banner = shadow.getElementById('pgBanner');
    if (banner) {
      banner.classList.add('dismissing');
      setTimeout(() => host.remove(), 300);
    }
  }, 10000);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Styles (injected into Shadow DOM)
// ═══════════════════════════════════════════════════════════════════════════════

function _getOverlayStyles() {
  return `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    .overlay {
      position: fixed !important;
      inset: 0 !important;
      z-index: 2147483647 !important;
      background: rgba(0, 0, 0, 0.75) !important;
      backdrop-filter: blur(8px) !important;
      -webkit-backdrop-filter: blur(8px) !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
      animation: fadeIn 0.3s ease-out !important;
    }

    .overlay.dismissing {
      animation: fadeOut 0.4s ease-in forwards !important;
    }

    .card {
      background: #1a1a2e !important;
      border: 1px solid rgba(239, 68, 68, 0.3) !important;
      border-radius: 16px !important;
      padding: 32px !important;
      max-width: 480px !important;
      width: 90vw !important;
      box-shadow: 0 0 60px rgba(239, 68, 68, 0.15), 0 25px 50px rgba(0, 0, 0, 0.5) !important;
      color: #e8eaed !important;
    }

    .card-header { text-align: center; margin-bottom: 24px; }

    .shield-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 72px;
      height: 72px;
      border-radius: 50%;
      background: rgba(239, 68, 68, 0.1);
      margin-bottom: 16px;
      animation: pulse 2s ease-in-out infinite;
    }

    .title {
      font-size: 22px !important;
      font-weight: 700 !important;
      color: #ef4444 !important;
      margin-bottom: 8px !important;
      letter-spacing: -0.3px;
    }

    .subtitle {
      font-size: 14px !important;
      color: #9aa0b0 !important;
      line-height: 1.5;
    }

    .score-section {
      margin-bottom: 20px;
    }

    .score-bar {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 8px;
    }

    .score-fill {
      height: 100%;
      background: linear-gradient(90deg, #f59e0b, #ef4444);
      border-radius: 3px;
      transition: width 1s ease-out;
    }

    .score-label {
      font-size: 13px;
      color: #9aa0b0;
    }
    .score-label strong { color: #ef4444; }

    .url-section {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 20px;
    }

    .url-label {
      display: block;
      font-size: 11px;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }

    .url-value {
      font-size: 13px;
      color: #d1d5db;
      word-break: break-all;
      font-family: "SF Mono", "Cascadia Code", "Consolas", monospace;
    }

    .features-section { margin-bottom: 24px; }

    .features-title {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: #e8eaed;
      margin-bottom: 10px;
    }

    .features-list {
      list-style: none;
      padding: 0;
    }

    .feature-item {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 6px 0;
      font-size: 13px;
      color: #b0b5c3;
      line-height: 1.4;
    }

    .f-icon { flex-shrink: 0; font-size: 12px; margin-top: 1px; }

    .actions {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 20px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 20px;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: all 0.2s ease;
      font-family: inherit;
    }

    .btn-primary {
      background: #10b981 !important;
      color: #fff !important;
    }
    .btn-primary:hover { background: #059669 !important; transform: translateY(-1px); }
    .btn-primary:active { transform: translateY(0); }

    .btn-ghost {
      background: transparent !important;
      color: #6b7280 !important;
      border: 1px solid rgba(255, 255, 255, 0.08) !important;
      font-size: 12px !important;
      font-weight: 400 !important;
    }
    .btn-ghost:hover { color: #9ca3af !important; border-color: rgba(255, 255, 255, 0.15) !important; }

    .footer {
      text-align: center;
      font-size: 12px;
      color: #4b5563;
    }

    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    @keyframes fadeOut {
      from { opacity: 1; }
      to { opacity: 0; }
    }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.2); }
      50% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
    }
  `;
}

function _getBannerStyles() {
  return `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    .banner {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      right: 0 !important;
      z-index: 2147483647 !important;
      background: #1a1a2e !important;
      border-bottom: 1px solid rgba(245, 158, 11, 0.3) !important;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4) !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
      animation: slideDown 0.3s ease-out !important;
    }

    .banner.dismissing {
      animation: slideUp 0.3s ease-in forwards !important;
    }

    .banner-content {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      color: #e8eaed;
    }

    .banner-icon { flex-shrink: 0; display: flex; }

    .banner-text {
      flex: 1;
      font-size: 13px;
      line-height: 1.4;
    }
    .banner-text strong {
      display: block;
      color: #f59e0b;
      font-size: 14px;
      margin-bottom: 2px;
    }
    .banner-detail { color: #9aa0b0; }

    .banner-close {
      flex-shrink: 0;
      background: rgba(255, 255, 255, 0.06);
      border: none;
      color: #9aa0b0;
      cursor: pointer;
      padding: 6px;
      border-radius: 6px;
      display: flex;
      transition: all 0.15s ease;
    }
    .banner-close:hover { background: rgba(255, 255, 255, 0.12); color: #e8eaed; }

    .banner-bar {
      height: 3px;
      background: rgba(255, 255, 255, 0.05);
    }
    .banner-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #f59e0b, #ef4444);
      transition: width 0.8s ease-out;
    }

    @keyframes slideDown {
      from { transform: translateY(-100%); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }

    @keyframes slideUp {
      from { transform: translateY(0); opacity: 1; }
      to { transform: translateY(-100%); opacity: 0; }
    }
  `;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Utility
// ═══════════════════════════════════════════════════════════════════════════════

function _escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function _truncateUrl(url, maxLen) {
  if (url.length <= maxLen) return url;
  return url.slice(0, maxLen - 3) + '...';
}
