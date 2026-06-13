/**
 * PhishGuard V2 — Popup Logic
 * popup/popup.js
 *
 * Drives the popup UI:
 *   1. Fetch the current tab's scan result from the service worker
 *   2. Animate the risk gauge to the score value
 *   3. Render verdict, URL, features, and scan metadata
 *   4. Handle re-scan and report actions
 *
 * All communication happens via chrome.runtime.sendMessage.
 */

'use strict';

// ─── DOM References ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const DOM = {
  loadingState:   $('loadingState'),
  mainView:       $('mainView'),
  gaugeFill:      $('gaugeFill'),
  gaugeScore:     $('gaugeScore'),
  verdictBadge:   $('verdictBadge'),
  currentUrl:     $('currentUrl'),
  statusDot:      $('statusDot'),
  scanDuration:   $('scanDuration'),
  urlScoreChip:   $('urlScoreChip'),
  domScoreChip:   $('domScoreChip'),
  featuresList:   $('featuresList'),
  featureCount:   $('featureCount'),
  featuresSection:$('featuresSection'),
  rescanBtn:      $('rescanBtn'),
  detailsBtn:     $('detailsBtn'),
  reportBtn:      $('reportBtn'),
  gradStop1:      $('gradStop1'),
  gradStop2:      $('gradStop2'),
  aiWarningSection: $('aiWarningSection'),
  aiWarningText:    $('aiWarningText'),
};

// ─── Constants ────────────────────────────────────────────────────────────────
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // r = 52

const VERDICT_CONFIG = {
  safe:       { label: 'Safe',        className: 'safe',       dotColor: '#10b981', grad1: '#10b981', grad2: '#34d399' },
  suspicious: { label: 'Suspicious',  className: 'suspicious', dotColor: '#f59e0b', grad1: '#f59e0b', grad2: '#fbbf24' },
  phishing:   { label: 'Phishing',    className: 'phishing',   dotColor: '#ef4444', grad1: '#ef4444', grad2: '#f87171' },
  scanning:   { label: 'Scanning...', className: 'scanning',   dotColor: '#6366f1', grad1: '#6366f1', grad2: '#818cf8' },
};


// ═══════════════════════════════════════════════════════════════════════════════
//  Initialisation
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const result = await getResult();
    if (result && result.verdict) {
      showMainView(result);
    } else {
      showNoData();
    }
  } catch (err) {
    console.error('[PhishGuard Popup] Init error:', err);
    showNoData();
  }

  // Wire up buttons
  DOM.rescanBtn.addEventListener('click', handleRescan);
  DOM.reportBtn.addEventListener('click', handleReport);
  DOM.detailsBtn.addEventListener('click', handleDetails);
});


// ═══════════════════════════════════════════════════════════════════════════════
//  Data Fetching
// ═══════════════════════════════════════════════════════════════════════════════

/** Request the current tab's scan result from the service worker. */
function getResult() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: 'GET_RESULT' }, (response) => {
      if (chrome.runtime.lastError) {
        resolve(null);
        return;
      }
      resolve(response);
    });
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
//  View Rendering
// ═══════════════════════════════════════════════════════════════════════════════

function showMainView(result) {
  DOM.loadingState.style.display = 'none';
  DOM.mainView.style.display = 'flex';

  animateGauge(result.score, result.verdict);
  setVerdict(result.verdict);
  setUrl(result.url);
  setScanInfo(result);
  
  if (result.isAiReport && result.socWarning) {
    DOM.aiWarningSection.style.display = 'block';
    DOM.aiWarningText.textContent = result.socWarning;
  } else {
    DOM.aiWarningSection.style.display = 'none';
  }

  renderFeatures(result.features || []);
}

function showNoData() {
  DOM.loadingState.style.display = 'none';
  DOM.mainView.style.display = 'flex';

  DOM.gaugeScore.textContent = '--';
  setVerdict('scanning');
  DOM.currentUrl.textContent = 'Navigate to a page to scan';
  DOM.featuresList.innerHTML = `
    <div class="features-empty">
      <div class="empty-icon">🔍</div>
      No scan data yet. Visit a website to begin analysis.
    </div>
  `;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Gauge Animation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Animate the SVG circular gauge from 0 to the target score.
 * Simultaneously counts up the numeric display.
 */
function animateGauge(score, verdict) {
  const targetPct = Math.round(score * 100);
  const config = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.safe;

  // Set gradient colors
  DOM.gradStop1.setAttribute('stop-color', config.grad1);
  DOM.gradStop2.setAttribute('stop-color', config.grad2);

  // Set gauge glow color
  DOM.gaugeFill.style.filter = `drop-shadow(0 0 8px ${config.grad1}40)`;

  // Animate the stroke-dashoffset
  const targetOffset = GAUGE_CIRCUMFERENCE * (1 - score);
  requestAnimationFrame(() => {
    DOM.gaugeFill.style.strokeDashoffset = targetOffset;
  });

  // Animate the numeric counter
  animateCounter(DOM.gaugeScore, 0, targetPct, 1200);
}

/**
 * Smoothly count a number from start to end over duration ms.
 * Uses easeOutExpo for a satisfying deceleration.
 */
function animateCounter(element, start, end, duration) {
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // easeOutExpo curve
    const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);

    const current = Math.round(start + (end - start) * eased);
    element.textContent = current;

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  UI Updates
// ═══════════════════════════════════════════════════════════════════════════════

function setVerdict(verdict) {
  const config = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.safe;

  // Badge
  DOM.verdictBadge.textContent = config.label;
  DOM.verdictBadge.className = `verdict-text ${config.className}`;

  // Status dot
  DOM.statusDot.style.background = config.dotColor;
  DOM.statusDot.style.boxShadow = `0 0 8px ${config.dotColor}40`;
}

function setUrl(url) {
  if (!url) {
    DOM.currentUrl.textContent = '—';
    return;
  }

  // Truncate for display but keep full URL in title for hover
  const displayUrl = url.length > 55 ? url.slice(0, 52) + '...' : url;
  DOM.currentUrl.textContent = displayUrl;
  DOM.currentUrl.title = url;
}

function setScanInfo(result) {
  // Duration
  if (result.scanDuration != null) {
    DOM.scanDuration.textContent = `${result.scanDuration}ms`;
  }

  // URL score
  if (result.urlScore != null) {
    DOM.urlScoreChip.textContent = `${Math.round(result.urlScore * 100)}%`;
  }

  // DOM score
  if (result.domScore != null) {
    DOM.domScoreChip.textContent = `${Math.round(result.domScore * 100)}%`;
  } else {
    DOM.domScoreChip.textContent = 'N/A';
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Feature List Rendering
// ═══════════════════════════════════════════════════════════════════════════════

function renderFeatures(features) {
  const list = DOM.featuresList;
  list.innerHTML = '';

  if (!features || features.length === 0) {
    list.innerHTML = `
      <div class="features-empty">
        <div class="empty-icon">✅</div>
        No suspicious signals detected
      </div>
    `;
    DOM.featureCount.textContent = '0 signals';
    return;
  }

  DOM.featureCount.textContent = `${features.length} signal${features.length !== 1 ? 's' : ''}`;

  features.forEach((feature, index) => {
    const card = document.createElement('div');
    card.className = 'feature-card';
    card.style.animationDelay = `${index * 0.05}s`;

    const severity = feature.severity || _inferSeverity(feature.weight);

    const dot = document.createElement('div');
    dot.className = `feature-severity ${severity}`;

    const label = document.createElement('span');
    label.className = 'feature-label';
    label.textContent = feature.label || feature.name;

    card.appendChild(dot);
    card.appendChild(label);
    list.appendChild(card);
  });
}

function _inferSeverity(weight) {
  if (weight >= 3.0) return 'critical';
  if (weight >= 2.0) return 'high';
  if (weight >= 1.0) return 'medium';
  return 'low';
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Button Handlers
// ═══════════════════════════════════════════════════════════════════════════════

function handleRescan() {
  // Reset UI to scanning state
  DOM.gaugeScore.textContent = '--';
  DOM.gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  setVerdict('scanning');
  DOM.featuresList.innerHTML = '';

  // Trigger re-analysis via service worker
  chrome.runtime.sendMessage({ type: 'REANALYZE' }, () => {
    // Poll for new result after a short delay
    setTimeout(async () => {
      const result = await getResult();
      if (result && result.verdict) {
        showMainView(result);
      }
    }, 1500);
  });
}

function handleDetails() {
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    if (!tabs[0]) return;

    const result = await getResult();
    if (!result) return;

    // Visual feedback
    DOM.detailsBtn.querySelector('.btn-icon').textContent = '✓';
    setTimeout(() => {
      DOM.detailsBtn.querySelector('.btn-icon').textContent = '📊';
    }, 1500);

    // Store the full result and open the details page in a new tab
    chrome.storage.local.set({ lastDetailedResult: result }, () => {
      chrome.tabs.create({ url: chrome.runtime.getURL('details/details.html') });
    });
  });
}

function handleReport() {
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    if (!tabs[0]) return;

    // Visual feedback
    DOM.reportBtn.querySelector('.btn-icon').textContent = '✓';
    DOM.reportBtn.style.pointerEvents = 'none';

    setTimeout(() => {
      DOM.reportBtn.querySelector('.btn-icon').textContent = '🚩';
      DOM.reportBtn.style.pointerEvents = '';
    }, 2000);

    // Get the current prediction to determine the correct label
    const result = await getResult();
    const isAiPhishing = result && result.verdict !== 'safe';
    // If AI predicted Phishing, the user is saying it's Safe (0)
    // If AI predicted Safe, the user is saying it's Phishing (1)
    const correctLabel = isAiPhishing ? 0 : 1;

    // Send Phase 5 Feedback to backend
    try {
      await fetch('http://localhost:8000/feedback/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: tabs[0].url,
          correct_label: correctLabel
        })
      });
      console.log('Report successfully sent to backend for retraining!');
    } catch (err) {
      console.error('Failed to send report to backend:', err);
    }
  });
}
