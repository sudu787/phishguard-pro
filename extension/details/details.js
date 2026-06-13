document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(['lastDetailedResult'], (data) => {
    const output = document.getElementById('jsonOutput');
    if (data.lastDetailedResult) {
      output.textContent = JSON.stringify(data.lastDetailedResult, null, 2);
    } else {
      output.textContent = 'No recent scan data found. Please open the PhishGuard extension on a website to generate a report.';
    }
  });

  document.getElementById('copyBtn').addEventListener('click', () => {
    const text = document.getElementById('jsonOutput').textContent;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('copyBtn');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy JSON'; }, 2000);
    });
  });
});
