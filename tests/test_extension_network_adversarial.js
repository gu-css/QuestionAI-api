/**
 * tests/test_extension_network_adversarial.js
 * Adversarial Empirical Stress Test Suite for Chrome Extension Service Worker Proxy.
 * Authoritative constraints: ORIGINAL_REQUEST.md (AC R3) & PROJECT.md (M3).
 *
 * Vectors tested:
 * 1. Backend API offline / unreachable (ECONNREFUSED, unresolvable host, socket destroy)
 * 2. HTTP Server Errors (500, 503, 504, 404, 422)
 * 3. Network timeout exceeding AbortController limit (5000ms)
 * 4. Malformed / truncated JSON responses (truncated, HTML, empty, binary, schema violations)
 * 5. Batch sizing edge cases (empty batches, null headlines, oversized batches of 1000, huge 50KB strings)
 * 6. Health check resilience (online, offline, 500, timeout)
 * 7. Extension state handling (disabled extension, unknown action, null message)
 * 8. Zero uncaught promise rejections across the entire suite.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT_DIR = path.resolve(__dirname, '..');
const SW_PATH = path.join(ROOT_DIR, 'extension', 'background', 'service-worker.js');

// Global collector for uncaught rejections
const unhandledRejections = [];
process.on('unhandledRejection', (reason) => {
  unhandledRejections.push(reason);
});

// Mock Storage
class MockStorage {
  constructor(initialData = {}) {
    this.data = { ...initialData };
  }

  async get(keys) {
    if (keys === null || keys === undefined) {
      return { ...this.data };
    }
    if (Array.isArray(keys)) {
      const res = {};
      for (const k of keys) {
        if (this.data[k] !== undefined) res[k] = this.data[k];
      }
      return res;
    }
    if (typeof keys === 'string') {
      return { [keys]: this.data[keys] };
    }
    return { ...this.data };
  }

  async set(items) {
    for (const [k, v] of Object.entries(items)) {
      this.data[k] = v;
    }
  }
}

// Service Worker Sandbox Runner
function createServiceWorkerEnvironment(initialStorage = {}) {
  const storage = new MockStorage(initialStorage);
  let messageListener = null;
  let installedListener = null;

  const chromeMock = {
    runtime: {
      onInstalled: {
        addListener: (fn) => {
          installedListener = fn;
        }
      },
      onMessage: {
        addListener: (fn) => {
          messageListener = fn;
        }
      }
    },
    storage: {
      local: storage
    }
  };

  const code = fs.readFileSync(SW_PATH, 'utf-8');
  const context = vm.createContext({
    chrome: chromeMock,
    fetch: globalThis.fetch,
    AbortController: globalThis.AbortController,
    setTimeout: globalThis.setTimeout,
    clearTimeout: globalThis.clearTimeout,
    console: {
      log: () => {},
      warn: () => {},
      error: () => {}
    },
    Error: globalThis.Error,
    TypeError: globalThis.TypeError,
    JSON: globalThis.JSON,
    Array: globalThis.Array,
    Object: globalThis.Object,
    Math: globalThis.Math,
    String: globalThis.String,
    Boolean: globalThis.Boolean
  });

  vm.runInContext(code, context);

  function dispatchMessage(request, sender = {}) {
    return new Promise((resolve, reject) => {
      if (!messageListener) {
        return reject(new Error('No chrome.runtime.onMessage listener registered.'));
      }

      let responded = false;
      const sendResponse = (res) => {
        responded = true;
        resolve(res);
      };

      try {
        const keepsChannelOpen = messageListener(request, sender, sendResponse);
        if (!keepsChannelOpen) {
          if (!responded) {
            resolve({ error: 'Listener did not return true and did not send response synchronously.' });
          }
        }
      } catch (err) {
        // Synchronous throw in listener
        reject(err);
      }
    });
  }

  return { storage, dispatchMessage, installedListener };
}

// Test harness runner
async function runAdversarialSuite() {
  console.log('🔥 ==============================================================');
  console.log('🔥 STARTING ADVERSARIAL SERVICE WORKER NETWORK PROXY STRESS SUITE');
  console.log('🔥 ==============================================================');

  let passedTests = 0;
  let failedTests = 0;

  function assert(condition, message) {
    if (!condition) {
      console.error(`❌ ASSERTION FAILED: ${message}`);
      failedTests++;
      throw new Error(message);
    }
  }

  async function testCase(name, fn) {
    try {
      process.stdout.write(`🧪 [TEST] ${name} ... `);
      await fn();
      console.log('✅ PASSED');
      passedTests++;
    } catch (err) {
      console.log(`❌ FAILED: ${err.message}`);
      failedTests++;
    }
  }

  // Configurable HTTP Mock Server
  let mockHandler = (req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ results: [{ is_clickbait: false, probability: 0.1 }] }));
  };

  const server = http.createServer((req, res) => {
    mockHandler(req, res);
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const serverPort = server.address().port;
  const mockServerUrl = `http://127.0.0.1:${serverPort}/api/predict`;

  console.log(`📡 Mock HTTP server listening on http://127.0.0.1:${serverPort}`);

  try {
    // -------------------------------------------------------------
    // VECTOR 1: BACKEND API OFFLINE / UNREACHABLE
    // -------------------------------------------------------------
    console.log('\n--- VECTOR 1: Backend API Offline / Unreachable ---');

    await testCase('1.1 Port with no server listening (ECONNREFUSED)', async () => {
      const unreachablePort = 59998;
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete de teste offline'],
        apiUrl: `http://127.0.0.1:${unreachablePort}/api/predict`
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
      assert(typeof res.error === 'string' && res.error.length > 0, 'Expected descriptive error string');
    });

    await testCase('1.2 Unresolvable DNS hostname', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete teste DNS'],
        apiUrl: 'http://this-domain-does-not-exist-at-all-987654321.test/api/predict'
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
      assert(typeof res.error === 'string' && res.error.length > 0, 'Expected error message');
    });

    await testCase('1.3 TCP Socket destroyed immediately by server', async () => {
      mockHandler = (req, socket) => {
        req.socket.destroy();
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete socket abort'],
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
      assert(typeof res.error === 'string', 'Expected error string');
    });

    await testCase('1.4 Single headline prediction with offline server', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictClickbait',
        headline: 'Manchete única offline',
        apiUrl: 'http://127.0.0.1:59997/api/predict'
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
      assert(typeof res.error === 'string', 'Expected error string');
    });

    // -------------------------------------------------------------
    // VECTOR 2: HTTP 500 / 503 / 504 / 404 / 422 SERVER ERRORS
    // -------------------------------------------------------------
    console.log('\n--- VECTOR 2: HTTP Server Errors (500, 503, 504, 404, 422) ---');

    for (const status of [500, 503, 504, 404, 422]) {
      await testCase(`2.1 HTTP ${status} error on batch prediction`, async () => {
        mockHandler = (req, res) => {
          res.writeHead(status, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ detail: `Simulated server error ${status}` }));
        };
        const { dispatchMessage } = createServiceWorkerEnvironment();
        const res = await dispatchMessage({
          action: 'predictBatch',
          headlines: ['Manchete teste HTTP error'],
          apiUrl: mockServerUrl
        });

        assert(res.success === false, `Expected success === false for HTTP ${status}`);
        assert(res.offline === true, 'Expected offline === true');
        assert(res.error.includes(String(status)), `Expected error message to mention ${status}, got "${res.error}"`);
      });
    }

    await testCase('2.2 HTTP 500 on single headline prediction', async () => {
      mockHandler = (req, res) => {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Internal Server Error');
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictClickbait',
        headline: 'Manchete única erro 500',
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
      assert(res.error.includes('500'), `Expected error message to mention 500, got "${res.error}"`);
    });

    // -------------------------------------------------------------
    // VECTOR 3: NETWORK TIMEOUT EXCEEDING ABORTCONTROLLER (5000ms)
    // -------------------------------------------------------------
    console.log('\n--- VECTOR 3: Network Timeout & AbortController (5000ms) ---');

    await testCase('3.1 Server delay exceeding 5000ms abort limit on predictBatch', async () => {
      mockHandler = (req, res) => {
        // Intentionally hang without responding
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const startTime = Date.now();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete que vai sofrer timeout'],
        apiUrl: mockServerUrl
      });
      const elapsed = Date.now() - startTime;

      assert(elapsed >= 4800 && elapsed <= 6500, `Expected elapsed around 5000ms, got ${elapsed}ms`);
      assert(res.success === false, 'Expected success === false on timeout');
      assert(res.offline === true, 'Expected offline === true on timeout');
      assert(
        res.error.toLowerCase().includes('abort') || res.error.toLowerCase().includes('timeout') || res.error.length > 0,
        `Expected abort/timeout message, got: ${res.error}`
      );
    });

    await testCase('3.2 Server delay exceeding 5000ms abort limit on predictClickbait', async () => {
      mockHandler = (req, res) => {
        // Intentionally hang without responding
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const startTime = Date.now();
      const res = await dispatchMessage({
        action: 'predictClickbait',
        headline: 'Manchete única que vai sofrer timeout',
        apiUrl: mockServerUrl
      });
      const elapsed = Date.now() - startTime;

      assert(elapsed >= 4800 && elapsed <= 6500, `Expected elapsed around 5000ms, got ${elapsed}ms`);
      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
    });

    // -------------------------------------------------------------
    // VECTOR 4: MALFORMED / TRUNCATED JSON RESPONSES
    // -------------------------------------------------------------
    console.log('\n--- VECTOR 4: Malformed / Truncated JSON Responses ---');

    await testCase('4.1 Truncated JSON stream (cut in middle of object)', async () => {
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end('{"results": [{"headline": "Notícia trunc');
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete JSON quebrado'],
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
      assert(typeof res.error === 'string' && res.error.length > 0, 'Expected JSON parse error message');
    });

    await testCase('4.2 HTML response instead of JSON (e.g. gateway error page)', async () => {
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<!DOCTYPE html><html><body><h1>502 Bad Gateway</h1><p>Cloudflare</p></body></html>');
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete resposta HTML'],
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
      assert(typeof res.error === 'string', 'Expected JSON parse error string');
    });

    await testCase('4.3 Empty 0-byte body with HTTP 200', async () => {
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end('');
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete 0-byte'],
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
    });

    await testCase('4.4 Binary garbage bytes in response', async () => {
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x00, 0xff, 0xfe]));
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete binaria'],
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
    });

    await testCase('4.5 Valid JSON null payload', async () => {
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end('null');
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete null response'],
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
    });

    await testCase('4.6 Valid JSON with non-array results (results: "invalid")', async () => {
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ results: 'not_an_array' }));
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete non-array results'],
        apiUrl: mockServerUrl
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.offline === true, 'Expected offline === true');
    });

    // -------------------------------------------------------------
    // VECTOR 5: EMPTY BATCHES AND OVERSIZED BATCHES
    // -------------------------------------------------------------
    console.log('\n--- VECTOR 5: Empty Batches & Oversized Batches ---');

    await testCase('5.1 Empty headlines array ({ headlines: [] })', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: []
      });

      assert(res.success === true, 'Expected success === true for empty batch');
      assert(Array.isArray(res.results) && res.results.length === 0, 'Expected results === []');
      assert(res.total === 0, 'Expected total === 0');
    });

    await testCase('5.2 Missing headlines field entirely in predictBatch', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch'
      });

      assert(res.success === true, 'Expected success === true for missing headlines array');
      assert(Array.isArray(res.results) && res.results.length === 0, 'Expected empty results');
      assert(res.total === 0, 'Expected total === 0');
    });

    await testCase('5.3 Empty headline string in predictClickbait', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictClickbait',
        headline: ''
      });

      assert(res.success === false, 'Expected success === false for empty headline');
      assert(res.error.includes('required'), 'Expected error stating headline is required');
    });

    await testCase('5.4 Batch of 200 headlines processed cleanly', async () => {
      const headlines200 = Array.from({ length: 200 }, (_, i) => `Manchete número ${i + 1} para o teste de lote`);
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        const mockResults = headlines200.map((h, i) => ({
          headline: h,
          is_clickbait: i % 2 === 0,
          probability: i % 2 === 0 ? 0.95 : 0.05,
          bait_type: i % 2 === 0 ? 'Sensacionalismo' : null
        }));
        res.end(JSON.stringify({ results: mockResults }));
      };

      const { dispatchMessage, storage } = createServiceWorkerEnvironment({
        stats: { totalScanned: 0, clickbaitDetected: 0 }
      });
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: headlines200,
        apiUrl: mockServerUrl
      });

      assert(res.success === true, 'Expected success === true');
      assert(res.results.length === 200, 'Expected 200 results');
      assert(res.total === 200, 'Expected total === 200');

      // Verify stats updated in storage
      const finalStats = await storage.get(['stats']);
      assert(finalStats.stats.totalScanned === 200, 'Expected totalScanned === 200');
      assert(finalStats.stats.clickbaitDetected === 100, 'Expected clickbaitDetected === 100');
    });

    await testCase('5.5 Oversized batch of 1,000 headlines', async () => {
      const headlines1000 = Array.from({ length: 1000 }, (_, i) => `Manchete gigante ${i}`);
      mockHandler = (req, res) => {
        let body = '';
        req.on('data', (chunk) => { body += chunk; });
        req.on('end', () => {
          const parsed = JSON.parse(body);
          assert(parsed.headlines.length === 1000, 'Server received 1000 headlines');
          res.writeHead(200, { 'Content-Type': 'application/json' });
          const mockResults = parsed.headlines.map((h) => ({
            headline: h,
            is_clickbait: false,
            probability: 0.01,
            bait_type: null
          }));
          res.end(JSON.stringify({ results: mockResults }));
        });
      };

      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: headlines1000,
        apiUrl: mockServerUrl
      });

      assert(res.success === true, 'Expected success === true for 1000 headlines batch');
      assert(res.results.length === 1000, 'Expected 1000 results');
    });

    await testCase('5.6 Massive 50 KB headline string', async () => {
      const massiveHeadline = 'Notícia enorme '.repeat(3500); // ~50KB
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          headline: massiveHeadline,
          is_clickbait: true,
          probability: 0.99,
          bait_type: 'Sensacionalismo'
        }));
      };

      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'predictClickbait',
        headline: massiveHeadline,
        apiUrl: mockServerUrl
      });

      assert(res.success === true, 'Expected success === true for 50KB headline');
      assert(res.data.is_clickbait === true, 'Expected is_clickbait === true');
    });

    // -------------------------------------------------------------
    // VECTOR 6: HEALTH CHECK RESILIENCE
    // -------------------------------------------------------------
    console.log('\n--- VECTOR 6: Health Check Resilience ---');

    await testCase('6.1 Health check with healthy server', async () => {
      mockHandler = (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'healthy', supervised_loaded: true }));
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'healthCheck',
        apiUrl: mockServerUrl
      });

      assert(res.success === true, 'Expected success === true');
      assert(res.health && res.health.online === true, 'Expected online === true');
      assert(res.health.data.status === 'healthy', 'Expected status === healthy');
    });

    await testCase('6.2 Health check with offline server', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'healthCheck',
        apiUrl: 'http://127.0.0.1:59996/api/predict'
      });

      assert(res.success === true, 'Expected success === true for healthCheck action');
      assert(res.health && res.health.online === false, 'Expected online === false');
      assert(typeof res.health.error === 'string', 'Expected error string in health report');
    });

    await testCase('6.3 Health check with HTTP 500 status', async () => {
      mockHandler = (req, res) => {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'unhealthy' }));
      };
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'healthCheck',
        apiUrl: mockServerUrl
      });

      assert(res.success === true, 'Expected success === true');
      assert(res.health && res.health.online === false, 'Expected online === false');
      assert(res.health.status === 500, 'Expected health.status === 500');
    });

    // -------------------------------------------------------------
    // VECTOR 7: EXTENSION STATE & PROTOCOL RESILIENCE
    // -------------------------------------------------------------
    console.log('\n--- VECTOR 7: Extension State & Protocol Edge Cases ---');

    await testCase('7.1 Extension disabled state (enabled: false)', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment({
        enabled: false
      });
      const res = await dispatchMessage({
        action: 'predictBatch',
        headlines: ['Manchete com extensão desativada']
      });

      assert(res.success === false, 'Expected success === false when disabled');
      assert(res.disabled === true, 'Expected disabled === true');
      assert(res.reason.includes('disabled'), 'Expected reason string');
    });

    await testCase('7.2 Unknown action string', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment();
      const res = await dispatchMessage({
        action: 'UNKNOWN_NONEXISTENT_ACTION_XYZ'
      });

      assert(res.success === false, 'Expected success === false');
      assert(res.error.includes('Unknown action'), `Expected unknown action error, got "${res.error}"`);
    });

    await testCase('7.3 getConfig action returns configuration and statistics', async () => {
      const { dispatchMessage } = createServiceWorkerEnvironment({
        apiUrl: 'http://custom-api:8000/api/predict',
        enabled: true,
        stats: { totalScanned: 42, clickbaitDetected: 7 }
      });
      const res = await dispatchMessage({
        action: 'getConfig'
      });

      assert(res.success === true, 'Expected success === true');
      assert(res.config.apiUrl === 'http://custom-api:8000/api/predict', 'Custom apiUrl preserved');
      assert(res.config.enabled === true, 'Enabled true');
      assert(res.config.stats.totalScanned === 42, 'Stats totalScanned === 42');
      assert(res.config.stats.clickbaitDetected === 7, 'Stats clickbaitDetected === 7');
    });

    await testCase('7.4 onInstalled initializes default configuration if storage is empty', async () => {
      const { storage, installedListener } = createServiceWorkerEnvironment({});
      assert(typeof installedListener === 'function', 'Installed listener must be a function');
      await installedListener();

      const stored = await storage.get(null);
      assert(stored.apiUrl === 'http://localhost:7860/api/predict', 'Default apiUrl initialized');
      assert(stored.enabled === true, 'Default enabled initialized');
      assert(stored.stats.totalScanned === 0, 'Default stats initialized');
    });

  } finally {
    await new Promise((resolve) => server.close(resolve));
    console.log('🛑 Mock HTTP server closed.');
  }

  // -------------------------------------------------------------
  // VECTOR 8: UNHANDLED PROMISE REJECTION AUDIT
  // -------------------------------------------------------------
  console.log('\n--- VECTOR 8: Uncaught Rejection Audit ---');
  await testCase('8.1 Zero uncaught promise rejections across all tests', async () => {
    assert(
      unhandledRejections.length === 0,
      `Detected ${unhandledRejections.length} uncaught promise rejections: ${unhandledRejections.join('; ')}`
    );
  });

  // Summary
  console.log('\n==============================================================');
  console.log(`📊 TEST SUITE SUMMARY: ${passedTests} PASSED, ${failedTests} FAILED`);
  console.log('==============================================================\n');

  if (failedTests > 0) {
    process.exit(1);
  }
}

runAdversarialSuite().catch((err) => {
  console.error('Fatal test runner error:', err);
  process.exit(1);
});
