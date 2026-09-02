// Headless end-to-end verification of the GROKVERSE explorer (PROMPT.md §3).
// Drives real Chromium: confirms the 3D scene renders, real run data loads, the
// in-browser LiveLab actually groks, and the guided tour steps through.
// Usage: node verify.mjs   (dev server must be running on $URL, default :3000)
import { chromium } from "playwright";

const URL = process.env.URL || "http://localhost:3000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fail = (m) => { console.error("FAIL:", m); process.exit(1); };
const ok = (m) => console.log("ok:", m);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const pageErrors = [];
const consoleErrors = [];
page.on("pageerror", (e) => pageErrors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });

await page.goto(URL, { waitUntil: "networkidle", timeout: 60000 });

await page.waitForSelector("canvas", { timeout: 20000 });
ok("3D canvas rendered");

await page.waitForFunction(() => document.querySelectorAll("select option").length > 0, { timeout: 20000 });
const runCount = await page.$$eval("select option", (os) => os.length);
if (runCount < 1) fail("no runs in ControlPanel");
ok(`ControlPanel lists ${runCount} runs`);

// Progress-measure overlay: the panel must render exactly for runs whose meta
// carries the measured restricted/excluded-loss curves, and never for runs
// without them (PROMPT.md §6: no fabricated data for runs missing the measure).
const index = await (await fetch(`${URL}/data/index.json`)).json();
const metas = await Promise.all(index.map(async (r) => {
  const meta = await (await fetch(`${URL}/data/${r.id}.meta.json`)).json();
  return { id: r.id, hasPM: Boolean(meta.progress_measures) };
}));
const pmProbes = [metas.find((m) => m.hasPM), metas.find((m) => !m.hasPM)].filter(Boolean);
if (!metas.some((m) => m.hasPM)) fail("no exported run has progress_measures — expected at least one");
for (const probe of pmProbes) {
  await page.selectOption("select", probe.id);
  await page.waitForFunction(
    (want) => document.querySelector("input[type=range]") !== null &&
      (document.querySelector("[data-testid=progress-measures]") !== null) === want,
    probe.hasPM, { timeout: 20000 },
  );
  ok(`progress-measures panel ${probe.hasPM ? "shown" : "absent"} for ${probe.id}`);
}
await page.selectOption("select", metas[0].id);
await page.waitForFunction(() => document.querySelector("input[type=range]") !== null, { timeout: 20000 });

// Live Lab: open, wait for tfjs, train, assert grokking
await page.click("[data-testid=toggle-lab]");
await page.waitForSelector("[data-testid=livelab-toggle]", { timeout: 20000 });
await page.waitForFunction(() => {
  const b = document.querySelector("[data-testid=livelab-toggle]");
  return b && !b.disabled;
}, { timeout: 60000 });
ok("LiveLab tfjs backend loaded");
await page.click("[data-testid=livelab-toggle]");
let s = null;
for (let t = 0; t < 160; t++) {
  await sleep(500);
  s = await page.evaluate(() => window.__livelab || null);
  if (s && s.testAcc > 0.9 && s.trainAcc > 0.95) break;
}
if (!s) fail("LiveLab produced no metrics");
console.log("  livelab final:", JSON.stringify(s));
if (!(s.testAcc > 0.9 && s.trainAcc > 0.95)) {
  fail(`LiveLab did not grok (train ${s.trainAcc.toFixed(3)}, test ${s.testAcc.toFixed(3)}, step ${s.step})`);
}
// "Grokked" means DELAYED generalization: test accuracy must have arrived
// well after train saturated (same >= 100-step bar as LiveLab's MIN_GROK_DELAY;
// measured gaps at the default sliders run 1300+).
if (!(s.grokGap != null && s.grokGap >= 100)) {
  fail(`LiveLab generalized without a grok delay (trainSat ${s.trainSatStep}, testGen ${s.testGenStep})`);
}
ok(`LiveLab GROKKED: train ${s.trainAcc.toFixed(3)} test ${s.testAcc.toFixed(3)} @ step ${s.step} (grok gap ${s.grokGap})`);

await page.click("[data-testid=toggle-lab]"); // back to explorer

// Guided tour: start + step all the way through
await page.click("[data-testid=start-tour]");
await page.waitForSelector("[data-testid=guided-tour]", { timeout: 10000 });
for (let i = 0; i < 8; i++) {
  await sleep(250);
  const btn = await page.$("[data-testid=tour-next]");
  if (!btn) break;
  await btn.click();
}
ok("guided tour stepped through");

await page.screenshot({ path: "verify-explorer.png" });
ok("screenshot saved -> verify-explorer.png");

if (pageErrors.length) {
  console.error("PAGE ERRORS:", pageErrors.slice(0, 5));
  fail(`${pageErrors.length} uncaught page errors`);
}
if (consoleErrors.length) {
  // Phase 6 acceptance is "no console errors" — enforce it, don't downgrade it.
  console.error("CONSOLE ERRORS:", consoleErrors.slice(0, 5));
  fail(`${consoleErrors.length} console errors`);
}

console.log("ALL WEB CHECKS PASSED");
await browser.close();
process.exit(0);
