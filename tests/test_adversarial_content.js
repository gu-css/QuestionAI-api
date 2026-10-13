/**
 * tests/test_adversarial_content.js
 * Empirical Adversarial Test Harness for Chrome Extension Content Script (extension/content/content.js).
 *
 * Verifies edge cases, adversarial HTML, rapid mutation bursts, infinite loop prevention,
 * XSS escaping, boilerplate rejection, and idempotency.
 */

const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');

// Repos are split: extension/ is a sibling of api/, old monorepo layout as fallback.
const _LOCAL_CONTENT = path.resolve(__dirname, '../extension/content/content.js');
const _SIBLING_CONTENT = path.resolve(__dirname, '../../extension/content/content.js');
const CONTENT_SCRIPT_PATH = fs.existsSync(_SIBLING_CONTENT) ? _SIBLING_CONTENT : _LOCAL_CONTENT;
const contentScriptCode = fs.readFileSync(CONTENT_SCRIPT_PATH, 'utf8');

// Test utilities
function createMockEnvironment(initialHtml, mockApiResponse) {
  const dom = new JSDOM(initialHtml, { runScripts: 'dangerously' });
  const window = dom.window;
  const document = window.document;

  const messagesSent = [];
  let storageCallback = null;

  const chromeMock = {
    runtime: {
      sendMessage: (payload, callback) => {
        messagesSent.push(payload);
        if (callback) {
          if (typeof mockApiResponse === 'function') {
            callback(mockApiResponse(payload));
          } else if (mockApiResponse !== undefined) {
            callback(mockApiResponse);
          } else {
            // Default mock: mark all as clickbait
            const results = (payload.headlines || []).map((hl, i) => ({
              headline: hl,
              is_clickbait: true,
              clickbait_status: 'clickbait',
              probability: 0.95,
              confidence: 0.90,
              bait_type: 'Sensacionalismo / Exagero'
            }));
            callback({ success: true, results });
          }
        }
      }
    },
    storage: {
      onChanged: {
        addListener: (cb) => {
          storageCallback = cb;
        }
      }
    }
  };

  // Provide browser globals
  window.chrome = chromeMock;
  window.requestAnimationFrame = (cb) => setTimeout(cb, 0);
  window.cancelAnimationFrame = (id) => clearTimeout(id);
  global.window = window;
  global.document = document;
  global.Node = window.Node;
  global.MutationObserver = window.MutationObserver;
  global.requestAnimationFrame = window.requestAnimationFrame;
  global.cancelAnimationFrame = window.cancelAnimationFrame;
  global.chrome = chromeMock;

  return {
    dom,
    window,
    document,
    messagesSent,
    triggerStorageChange: (enabled) => {
      if (storageCallback) {
        storageCallback({ enabled: { newValue: enabled } }, 'local');
      }
    },
    runScript: () => {
      window.eval(contentScriptCode);
    }
  };
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runAdversarialTests() {
  console.log('='.repeat(70));
  console.log('  🧪 ADVERSARIAL STRESS TEST HARNESS — CHROME EXTENSION CONTENT SCRIPT');
  console.log('='.repeat(70));

  let passed = 0;
  let failed = 0;
  const failures = [];

  function assert(condition, message, details = '') {
    if (condition) {
      console.log(`  ✅ PASS: ${message}`);
      passed++;
    } else {
      console.error(`  ❌ FAIL: ${message}`);
      if (details) console.error(`     Details: ${details}`);
      failed++;
      failures.push({ message, details });
    }
  }

  // =========================================================================
  // SUITE 1: Nested headings inside links and cards
  // =========================================================================
  console.log('\n--- Suite 1: Nested Headings Inside Links and Cards ---');

  // Test 1.1: Link wrapping heading: <article><a href="..."><h2>Headline</h2></a></article>
  {
    const html = `
      <article>
        <a href="/news/1">
          <h2>Você não vai acreditar no que aconteceu com esse famoso hoje!</h2>
        </a>
      </article>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badges = env.document.querySelectorAll('.clickbait-badge');
    const detected = env.document.querySelectorAll('.clickbait-detected');
    assert(badges.length === 1, 'Link wrapping heading produces exactly 1 badge', `Got ${badges.length} badges`);
    assert(detected.length === 1, 'Link wrapping heading marks exactly 1 element detected', `Got ${detected.length}`);
  }

  // Test 1.2: Heading wrapping link: <article><h2><a href="...">Headline</a></h2></article>
  {
    const html = `
      <article>
        <h2>
          <a href="/news/2">Você não vai acreditar no que aconteceu com esse famoso hoje!</a>
        </h2>
      </article>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badges = env.document.querySelectorAll('.clickbait-badge');
    const detected = env.document.querySelectorAll('.clickbait-detected');
    assert(badges.length === 1, 'Heading wrapping link produces exactly 1 badge (no duplicate badge on a and h2)', `Got ${badges.length} badges`);
    assert(detected.length === 1, 'Heading wrapping link marks exactly 1 element detected', `Got ${detected.length}`);
  }

  // Test 1.3: Card container with .headline class wrapping h3
  {
    const html = `
      <div class="card headline">
        <h3>Revelado o grande mistério sobre o caso do momento imperdível</h3>
      </div>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badges = env.document.querySelectorAll('.clickbait-badge');
    assert(badges.length === 1, 'Card container .headline wrapping h3 produces exactly 1 badge', `Got ${badges.length} badges`);
  }

  // Test 1.4: Card container with .title wrapping a
  {
    const html = `
      <div class="title">
        <a href="/news/3">Segredo revelado finalmente após anos de silêncio absoluto</a>
      </div>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badges = env.document.querySelectorAll('.clickbait-badge');
    assert(badges.length <= 1, 'Container .title with link produces at most 1 badge', `Got ${badges.length} badges`);
  }

  // =========================================================================
  // SUITE 2: Empty or whitespace-only headings & length boundaries
  // =========================================================================
  console.log('\n--- Suite 2: Empty or Whitespace-only Headings & Length Boundaries ---');

  {
    const html = `
      <div>
        <h1></h1>
        <h2>   </h2>
        <h3>\t\r\n</h3>
        <h4>&nbsp;&nbsp;&nbsp;</h4>
        <h1><span>   </span></h1>
        <h2><img src="tracker.gif" /></h2>
        <!-- Short headlines (<= 18 chars or <= 3 words) -->
        <h1>BOMBASTICO</h1>
        <h2>ATENÇÃO AGORA</h2>
        <h3>FIM DA LINHA</h3>
        <h4>Oi lá bom dia</h4>
        <h5>Inacreditavelmente Extraordinário Sensacional</h5>
      </div>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badges = env.document.querySelectorAll('.clickbait-badge');
    assert(badges.length === 0, 'Empty, whitespace, and short headlines produce 0 badges', `Got ${badges.length} badges`);
    assert(env.messagesSent.length === 0, 'No API messages sent for noise headlines', `Sent ${env.messagesSent.length} messages`);
  }

  // Boundary check: 18 chars vs 19 chars with 4 words
  {
    const html = `
      <div>
        <h2 id="h-short">123 567 901 abcd</h2>
        <h2 id="h-long">1234 6789 abcd efgh</h2>
      </div>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const hShortBadge = env.document.querySelector('#h-short .clickbait-badge');
    const hLongBadge = env.document.querySelector('#h-long .clickbait-badge');
    assert(hShortBadge === null, '17-char headline is rejected (<= 18)', 'Badge found on short');
    assert(hLongBadge !== null, '19-char 4-word headline is accepted (> 18 and > 3 words)', 'No badge on long');
  }

  // =========================================================================
  // SUITE 3: Massive headlines (>1000 characters & extreme DOM depth)
  // =========================================================================
  console.log('\n--- Suite 3: Massive Headlines (>1000 chars & deep DOM) ---');

  {
    const hugeText = 'Esta é uma manchete absolutamente gigantesca sobre acontecimentos impressionantes '.repeat(25); // ~2000 chars
    const html = `
      <article>
        <h1 id="huge">${hugeText}</h1>
      </article>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badge = env.document.querySelector('#huge .clickbait-badge');
    assert(badge !== null, 'Massive headline (>1000 chars) is processed and badged safely', 'Badge missing');
    assert(env.messagesSent[0].headlines[0].length >= 1000, 'Full massive text extracted cleanly', `Extracted length: ${env.messagesSent[0]?.headlines[0]?.length}`);
  }

  // Deep DOM nesting: 50 nested spans
  {
    let nested = 'Manchete com aninhamento profundo de elementos DOM!';
    for (let i = 0; i < 50; i++) {
      nested = `<span>${nested}</span>`;
    }
    const html = `<article><h2 id="deep">${nested}</h2></article>`;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badge = env.document.querySelector('#deep .clickbait-badge');
    assert(badge !== null, 'Deeply nested heading (50 spans) processed without recursion limit', 'Badge missing');
  }

  // =========================================================================
  // SUITE 4: Headlines with HTML entities, Unicode symbols, emojis, and XSS
  // =========================================================================
  console.log('\n--- Suite 4: HTML Entities, Unicode, Emojis, and XSS Resilience ---');

  {
    const headline = '🔥💣 &quot;ATENÇÃO&quot;: &#39;Você não vai acreditar&#39; &amp; &lt;Chocante&gt; 😱 🤔';
    const html = `<article><h2 id="unicode">${headline}</h2></article>`;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const sent = env.messagesSent[0]?.headlines[0];
    assert(sent.includes('🔥') && sent.includes('😱'), 'Emojis and symbols preserved in extracted headline', sent);
    assert(!sent.includes('&quot;'), 'HTML entities properly decoded by DOM textContent', sent);
  }

  // XSS Injection in API Response
  {
    const html = `<article><h2>Manchete para teste de injeção XSS maliciosa no selo</h2></article>`;
    const maliciousApiMock = {
      success: true,
      results: [
        {
          headline: 'Manchete para teste de injeção XSS maliciosa no selo',
          is_clickbait: true,
          bait_type: '<img src=x onerror=alert(1)><script>alert("xss")</script>',
          probability: 0.99,
          confidence: 0.95
        }
      ]
    };
    const env = createMockEnvironment(html, maliciousApiMock);
    env.runScript();
    await wait(450);

    const rawHtml = env.document.body.innerHTML;
    assert(!rawHtml.includes('<img src=x'), 'Malicious bait_type image tag is properly escaped (no <img)', rawHtml);
    assert(!rawHtml.includes('<script>'), 'Malicious bait_type script tag is properly escaped (no <script>)', rawHtml);
    assert(rawHtml.includes('&lt;img') || rawHtml.includes('&amp;lt;img'), 'Bait type contains escaped entities', rawHtml);
  }

  // Malformed API response data: NaN, null, undefined
  {
    const html = `<article><h2>Manchete para teste com dados numéricos nulos e NaN</h2></article>`;
    const oddApiMock = {
      success: true,
      results: [
        {
          headline: 'Manchete para teste com dados numéricos nulos e NaN',
          is_clickbait: true,
          bait_type: null,
          probability: NaN,
          confidence: null
        }
      ]
    };
    const env = createMockEnvironment(html, oddApiMock);
    let threw = false;
    try {
      env.runScript();
      await wait(450);
    } catch (e) {
      threw = true;
    }
    assert(!threw, 'Malformed API response (NaN, null) does not throw uncaught exceptions');
    const badge = env.document.querySelector('.clickbait-badge');
    assert(badge !== null, 'Badge injected with fallback values when fields are null/NaN');
  }

  // =========================================================================
  // SUITE 5: Navigation/Footer Headings That Must Be Ignored
  // =========================================================================
  console.log('\n--- Suite 5: Navigation, Footer, Header, and Aside Exclusions ---');

  {
    const html = `
      <div>
        <nav><h2>Links rápidos e seções recomendadas para navegação</h2></nav>
        <header><h1>Portal Principal de Notícias do Brasil e do Mundo</h1></header>
        <footer><h3>Termos de uso e privacidade da plataforma de notícias</h3></footer>
        <aside><h4>Leia também estas matérias especiais recomendadas</h4></aside>
        <button><h2>Clique aqui para assinar o jornal digital hoje</h2></button>
        <form><h3>Inscreva-se na nossa newsletter matinal agora mesmo</h3></form>
        <svg><foreignObject><h2>Texto decorativo vetorial não semântico do logotipo</h2></foreignObject></svg>
        <nav><div><ul><li><h3>Menu de esportes e entretenimento do portal</h3></li></ul></div></nav>
      </div>
    `;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const badges = env.document.querySelectorAll('.clickbait-badge');
    assert(badges.length === 0, 'All boilerplate parents (nav, header, footer, aside, button, form, svg) are ignored', `Found ${badges.length} badges`);
    assert(env.messagesSent.length === 0, 'No API messages sent for boilerplate elements', `Sent ${env.messagesSent.length}`);
  }

  // =========================================================================
  // SUITE 6: Rapid Burst Dynamic DOM Mutations & Infinite Mutation Loops
  // =========================================================================
  console.log('\n--- Suite 6: Rapid Mutation Bursts & Infinite Loop Prevention ---');

  {
    const html = `<div id="feed"><article><h2>Manchete inicial presente no carregamento da página</h2></article></div>`;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    const initialBadges = env.document.querySelectorAll('.clickbait-badge');
    assert(initialBadges.length === 1, 'Initial headline badged before burst');

    // Simulate rapid burst: 50 mutations in 50ms
    const feed = env.document.querySelector('#feed');
    for (let i = 1; i <= 20; i++) {
      const art = env.document.createElement('article');
      art.innerHTML = `<h3>Nova notícia dinâmica adicionada durante scroll infinito número ${i}!</h3>`;
      feed.appendChild(art);
    }

    // Wait for debounced scan (300ms + buffer)
    await wait(550);

    const totalBadges = env.document.querySelectorAll('.clickbait-badge');
    assert(totalBadges.length === 21, `All 20 dynamically injected headlines plus initial are badged (${totalBadges.length}/21)`, `Got ${totalBadges.length}`);
  }

  // Infinite Loop Test: Verify MutationObserver does not self-trigger on badge injection
  {
    const html = `<article><h2 id="loop-test">Manchete para verificar que injeção de badge não causa loop infinito</h2></article>`;
    const env = createMockEnvironment(html);

    let observerMutationEvents = 0;
    const originalObserver = env.window.MutationObserver;
    env.window.MutationObserver = function (callback) {
      return new originalObserver((mutations, obs) => {
        observerMutationEvents += mutations.length;
        callback(mutations, obs);
      });
    };

    env.runScript();
    await wait(600);

    const eventsAfterFirstScan = observerMutationEvents;
    // Wait an additional 500ms to verify zero further activity
    await wait(500);
    const eventsAfterIdle = observerMutationEvents;

    assert(eventsAfterIdle === eventsAfterFirstScan, 'System reaches steady-state with 0 infinite loop mutations', `Events before: ${eventsAfterFirstScan}, after: ${eventsAfterIdle}`);
  }

  // =========================================================================
  // SUITE 7: Idempotency & Storage Toggle
  // =========================================================================
  console.log('\n--- Suite 7: Idempotency & Storage Toggle ---');

  {
    const html = `<article><h2 id="idempotent">Você não vai acreditar no que aconteceu com esse famoso!</h2></article>`;
    const env = createMockEnvironment(html);
    env.runScript();
    await wait(450);

    let badges = env.document.querySelectorAll('.clickbait-badge');
    assert(badges.length === 1, 'Exactly 1 badge injected initially');

    // Trigger toggle off
    env.triggerStorageChange(false);
    badges = env.document.querySelectorAll('.clickbait-badge');
    const detected = env.document.querySelectorAll('.clickbait-detected');
    assert(badges.length === 0, 'Storage toggle false removes all badges', `Remaining: ${badges.length}`);
    assert(detected.length === 0, 'Storage toggle false removes all .clickbait-detected classes', `Remaining: ${detected.length}`);

    // Trigger toggle on
    env.triggerStorageChange(true);
    await wait(450);
    badges = env.document.querySelectorAll('.clickbait-badge');
    assert(badges.length === 1, 'Storage toggle true re-injects exactly 1 badge cleanly', `Got ${badges.length}`);

    // Trigger manual re-scan
    const textBefore = env.document.querySelector('#idempotent').textContent;
    // Run second scan
    env.window.eval('window.dispatchEvent(new Event("DOMContentLoaded"))');
    await wait(450);
    badges = env.document.querySelectorAll('.clickbait-badge');
    assert(badges.length === 1, 'Re-scan maintains exactly 1 badge (strict idempotency)', `Got ${badges.length}`);
  }

  // =========================================================================
  // SUMMARY
  // =========================================================================
  console.log('\n' + '='.repeat(70));
  console.log(` 📊 SUMMARY: ${passed} Passed, ${failed} Failed`);
  console.log('='.repeat(70));

  if (failed > 0) {
    console.error('\n❌ FAILURES RECORDED:');
    failures.forEach((f, idx) => {
      console.error(`  ${idx + 1}. ${f.message}`);
      if (f.details) console.error(`     ${f.details}`);
    });
    process.exit(1);
  } else {
    console.log('\n🎉 ALL ADVERSARIAL STRESS TESTS PASSED WITH 100% SUCCESS!');
    process.exit(0);
  }
}

runAdversarialTests().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});
