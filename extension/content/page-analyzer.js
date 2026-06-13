/**
 * PhishGuard V2 — Local Page Analyzer
 * content/page-analyzer.js
 *
 * Provides deep heuristic analysis of the DOM and scripts to detect
 * credential harvesting, hidden forms, and data exfiltration.
 * Runs locally to complement backend ML model.
 */

class PageAnalyzer {
  constructor() {
    this.results = {
      score: 0,
      signals: [],
      criticalThreat: false
    };
  }

  /**
   * Main entry point to analyze the current page.
   * @returns {Object} Analysis results
   */
  analyze() {
    this.results = { score: 0, signals: [], criticalThreat: false };
    
    this._checkFormsAndInputs();
    this._checkExfiltrationEndpoints();
    this._checkRedirects();
    this._checkHarvestingIndicators();
    
    return this.results;
  }

  /**
   * Helper to add a signal and update score.
   */
  _addSignal(name, severity, weight) {
    this.results.signals.push({ name, severity });
    this.results.score += weight;
    if (severity === 'critical') {
      this.results.criticalThreat = true;
    }
  }

  /**
   * 1 & 2: Login Forms, Password Fields, and Hidden Forms
   */
  _checkFormsAndInputs() {
    const forms = document.querySelectorAll('form');
    const passwordInputs = document.querySelectorAll('input[type="password"]');
    
    // Check for "floating" password inputs (no parent form) - common in JS stealers
    let floatingPasswords = 0;
    passwordInputs.forEach(input => {
      if (!input.closest('form')) floatingPasswords++;
    });
    
    if (floatingPasswords > 0) {
      this._addSignal('floating_password_input', 'high', 0.8);
    }

    // Check for hidden forms
    forms.forEach(form => {
      const style = window.getComputedStyle(form);
      const isHidden = (
        style.display === 'none' ||
        style.visibility === 'hidden' ||
        parseFloat(style.opacity) === 0 ||
        form.offsetWidth < 2 || form.offsetHeight < 2
      );
      
      if (isHidden) {
        // Only highly suspicious if there are also password inputs on the page
        if (passwordInputs.length > 0) {
           this._addSignal('hidden_form_with_passwords', 'high', 0.7);
        } else {
           this._addSignal('hidden_form', 'medium', 0.3);
        }
      }
    });
  }

  /**
   * 3: Data Exfiltration Endpoints
   */
  _checkExfiltrationEndpoints() {
    const forms = document.querySelectorAll('form');
    const scripts = document.querySelectorAll('script');
    
    // Known exfiltration patterns
    const exfilPatterns = [
      /discord\.com\/api\/webhooks/i,
      /api\.telegram\.org\/bot/i,
      /formspree\.io/i,
      /getform\.io/i,
      /formsubmit\.co/i
    ];

    // Check form actions
    forms.forEach(form => {
      const action = form.getAttribute('action') || '';
      if (exfilPatterns.some(p => p.test(action))) {
        this._addSignal('known_exfiltration_action', 'critical', 1.0);
      }
      
      // Check if action is cross-origin
      try {
        if (action && !action.startsWith('#') && !action.startsWith('javascript:')) {
           const actionUrl = new URL(action, location.href);
           if (actionUrl.hostname && actionUrl.hostname !== location.hostname) {
               // Posting credentials to another domain is suspicious
               if (document.querySelectorAll('input[type="password"]').length > 0) {
                   this._addSignal('cross_origin_credential_post', 'high', 0.6);
               }
           }
        }
      } catch (e) { /* ignore parse errors */ }
    });

    // Check script contents for hardcoded tokens
    scripts.forEach(script => {
      if (!script.src) {
        const content = script.textContent || '';
        if (exfilPatterns.some(p => p.test(content))) {
          this._addSignal('known_exfiltration_script', 'critical', 1.0);
        }
      }
    });
  }

  /**
   * 4: Suspicious Redirects
   */
  _checkRedirects() {
    const metaRefresh = document.querySelector('meta[http-equiv="refresh"]');
    if (metaRefresh) {
      const content = metaRefresh.getAttribute('content') || '';
      if (content.toLowerCase().includes('url=')) {
         this._addSignal('meta_refresh_redirect', 'medium', 0.4);
      }
    }
  }

  /**
   * 5: Credential Harvesting Indicators
   */
  _checkHarvestingIndicators() {
    const bodyText = (document.body?.innerText || '').substring(0, 5000).toLowerCase();
    
    // Generic login urgency
    if (/verify your account|session expired|login to continue|unusual activity/i.test(bodyText)) {
      if (document.querySelectorAll('input[type="password"]').length > 0) {
         this._addSignal('urgent_login_text', 'medium', 0.3);
      }
    }

    // Anti-analysis techniques
    const bodyContextMenu = document.body?.getAttribute('oncontextmenu') || '';
    if (/return\s*false/i.test(bodyContextMenu)) {
      this._addSignal('right_click_disabled', 'medium', 0.3);
    }
    
    const onKeyDown = document.body?.getAttribute('onkeydown') || '';
    if (onKeyDown.includes('123') || onKeyDown.includes('prevent')) {
       this._addSignal('dev_tools_blocked', 'high', 0.5);
    }
  }
}

// Make globally available in content script scope
window.PageAnalyzer = PageAnalyzer;
