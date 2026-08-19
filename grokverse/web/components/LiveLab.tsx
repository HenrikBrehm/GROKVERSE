"use client";

import type { CSSProperties } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

type Pt = { step: number; trainAcc: number; testAcc: number };

type TfModule = typeof import("@tensorflow/tfjs");

type LabState = {
  model: import("@tensorflow/tfjs").Sequential;
  opt: import("@tensorflow/tfjs").Optimizer;
  xTr: import("@tensorflow/tfjs").Tensor2D;
  yTr: import("@tensorflow/tfjs").Tensor1D;
  yTrOh: import("@tensorflow/tfjs").Tensor;
  xTe: import("@tensorflow/tfjs").Tensor2D;
  yTe: import("@tensorflow/tfjs").Tensor1D;
  step: number;
};

const P = 23; // small modulus: trains smoothly in-browser
const HIDDEN = 128;
const LR = 0.01;
const STEPS_PER_TICK = 25;
const MAX_HIST = 600; // beyond this the history is thinned 2:1, never truncated
// Train threshold matches the offline detector (train.py: 0.99). The test bar
// is DELIBERATELY relaxed to 0.9 (offline: 0.95): the tiny mod-23 in-browser
// MLP plateaus around ~0.91 in its step budget. verify.mjs uses the same 0.9.
const TRAIN_SAT_ACC = 0.99;
const TEST_GEN_ACC = 0.9;
// "Grokked" additionally requires this many steps of measured delay between
// memorization and generalization; verify.mjs asserts the same bar.
const MIN_GROK_DELAY = 100;

export function LiveLab() {
  const tf = useRef<TfModule | null>(null);
  const [ready, setReady] = useState(false);
  const [trainFrac, setTrainFrac] = useState(0.7);
  const [weightDecay, setWeightDecay] = useState(2);
  const wdRef = useRef(2);
  const [running, setRunning] = useState(false);
  const [hist, setHist] = useState<Pt[]>([]);
  const histRef = useRef<Pt[]>([]);
  const st = useRef<LabState | null>(null);
  const runningRef = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Grokking bookkeeping, independent of the (thinned) plot history.
  const trainSatStep = useRef<number | null>(null);
  const testGenStep = useRef<number | null>(null);

  useEffect(() => { wdRef.current = weightDecay; }, [weightDecay]);

  useEffect(() => {
    let on = true;
    import("@tensorflow/tfjs").then((m) => { if (on) { tf.current = m; setReady(true); } });
    return () => { on = false; };
  }, []);

  const dispose = () => {
    const s = st.current;
    if (!s) return;
    [s.xTr, s.yTr, s.yTrOh, s.xTe, s.yTe].forEach((t) => t?.dispose());
    s.model.dispose();
    s.opt.dispose(); // Adam moment tensors would otherwise leak on every rebuild
    st.current = null;
  };

  const build = useCallback(() => {
    const T = tf.current;
    if (!T) return;
    dispose();
    const n = P * P;
    const X: number[][] = [];
    const Y: number[] = [];
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const r = new Array(2 * P).fill(0);
      r[a] = 1; r[P + b] = 1;
      X.push(r); Y.push((a + b) % P);
    }
    let seed = 1234567;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const nTr = Math.max(1, Math.floor(trainFrac * n));
    const tr = idx.slice(0, nTr), te = idx.slice(nTr);
    const model = T.sequential();
    model.add(T.layers.dense({ inputShape: [2 * P], units: HIDDEN, activation: "relu" }));
    model.add(T.layers.dense({ units: P, activation: "linear" }));
    st.current = {
      model,
      opt: T.train.adam(LR),
      xTr: T.tensor2d(tr.map((i) => X[i])),
      yTr: T.tensor1d(tr.map((i) => Y[i]), "int32"),
      yTrOh: T.oneHot(T.tensor1d(tr.map((i) => Y[i]), "int32"), P),
      xTe: T.tensor2d(te.map((i) => X[i])),
      yTe: T.tensor1d(te.map((i) => Y[i]), "int32"),
      step: 0,
    };
    histRef.current = [];
    setHist([]);
    trainSatStep.current = null;
    testGenStep.current = null;
  }, [trainFrac]);

  useEffect(() => { if (ready && !runningRef.current) build(); }, [ready, build]);

  const accuracy = (model: LabState["model"], x: LabState["xTr"], y: LabState["yTr"]): number => {
    const T = tf.current!;
    return T.tidy(() =>
      (model.apply(x) as import("@tensorflow/tfjs").Tensor)
        .argMax(-1).equal(y).mean().dataSync()[0]);
  };

  const tick = useCallback(() => {
    const T = tf.current;
    const s = st.current;
    if (!T || !s || !runningRef.current) return;
    const factor = 1 - LR * wdRef.current;
    for (let i = 0; i < STEPS_PER_TICK; i++) {
      // minimize cross-entropy only ...
      s.opt.minimize(() => T.tidy(() =>
        T.losses.softmaxCrossEntropy(
          s.yTrOh, s.model.apply(s.xTr) as import("@tensorflow/tfjs").Tensor2D,
        ).mean() as import("@tensorflow/tfjs").Scalar
      ));
      // ... then decoupled (AdamW-style) weight decay, per step (matches the offline recipe)
      if (factor < 1) {
        T.tidy(() => { s.model.setWeights(s.model.getWeights().map((w) => w.mul(factor))); });
      }
      s.step++;
    }
    const trainAcc = accuracy(s.model, s.xTr, s.yTr);
    const testAcc = accuracy(s.model, s.xTe, s.yTe);
    if (trainSatStep.current == null && trainAcc >= TRAIN_SAT_ACC) trainSatStep.current = s.step;
    if (testGenStep.current == null && testAcc >= TEST_GEN_ACC) testGenStep.current = s.step;
    const grokGap = trainSatStep.current != null && testGenStep.current != null
      ? testGenStep.current - trainSatStep.current : null;
    (window as any).__livelab = {
      step: s.step, trainAcc, testAcc,
      trainSatStep: trainSatStep.current, testGenStep: testGenStep.current, grokGap,
    };
    histRef.current.push({ step: s.step, trainAcc, testAcc });
    if (histRef.current.length > MAX_HIST) {
      // thin 2:1 instead of discarding the head — the memorization plateau must
      // stay visible, it IS the grokking story
      histRef.current = histRef.current.filter((_, i) => i % 2 === 0);
    }
    if (s.step % (STEPS_PER_TICK * 6) < STEPS_PER_TICK) setHist([...histRef.current]);
    timer.current = setTimeout(tick, 0);
  }, []);

  const toggle = () => {
    if (runningRef.current) {
      runningRef.current = false; setRunning(false);
      if (timer.current) clearTimeout(timer.current);
      setHist([...histRef.current]);
    } else {
      if (!st.current) build();
      runningRef.current = true; setRunning(true);
      timer.current = setTimeout(tick, 0);
    }
  };
  const reset = () => {
    runningRef.current = false; setRunning(false);
    if (timer.current) clearTimeout(timer.current);
    build();
  };

  useEffect(() => () => {
    runningRef.current = false;
    if (timer.current) clearTimeout(timer.current);
    dispose();
  }, []);

  const cur = hist.length ? hist[hist.length - 1] : { step: 0, trainAcc: 0, testAcc: 0 };
  // "Grokked" is only claimed for DELAYED generalization — test accuracy arriving
  // well after train saturated. Immediate generalization is honest ordinary learning.
  const gap = trainSatStep.current != null && testGenStep.current != null
    ? testGenStep.current - trainSatStep.current : null;
  const status = gap != null && gap >= MIN_GROK_DELAY
    ? { text: `✓ grokked (gap ${gap})`, color: "#39d98a" }
    : gap != null
      ? { text: `generalized (gap ${gap} — no grok delay)`, color: "#7cb0ff" }
      : { text: "training…", color: "#8893a7" };

  return (
    <div style={wrap}>
      <div style={{ fontSize: 13, opacity: 0.82, lineHeight: 1.5 }}>
        In-browser MLP on <b>(a + b) mod {P}</b>. Raise weight decay and watch test
        accuracy suddenly snap up to meet train — grokking, live on your machine.
      </div>
      <MiniPlot hist={hist} />
      <div style={{ display: "flex", gap: 16, fontVariantNumeric: "tabular-nums", fontSize: 13 }}>
        <span data-testid="livelab-step">step {cur.step}</span>
        <span style={{ color: "#7cb0ff" }} data-testid="livelab-train-acc">train {cur.trainAcc.toFixed(3)}</span>
        <span style={{ color: "#ff8b8b" }} data-testid="livelab-test-acc">test {cur.testAcc.toFixed(3)}</span>
        <span data-testid="livelab-status" style={{ color: status.color }}>
          {status.text}
        </span>
      </div>
      <Slider label={`weight decay  ${weightDecay.toFixed(2)} (applies live)`} min={0} max={4} step={0.1}
        value={weightDecay} onChange={setWeightDecay} />
      <Slider
        label={`train fraction  ${trainFrac.toFixed(2)}${running ? " (pause to change)" : " (rebuilds the dataset)"}`}
        min={0.2} max={0.8} step={0.05}
        value={trainFrac} onChange={setTrainFrac} disabled={running} />
      <div style={{ display: "flex", gap: 10 }}>
        <button style={btn} onClick={toggle} disabled={!ready} data-testid="livelab-toggle">
          {running ? "❚❚ Pause" : ready ? "▶ Train" : "loading tf…"}
        </button>
        <button style={btn} onClick={reset} disabled={!ready}>↺ Reset</button>
      </div>
    </div>
  );
}

function MiniPlot({ hist }: { hist: Pt[] }) {
  const W = 368, H = 110, PAD = 8;
  if (hist.length < 2) {
    return <div style={{ height: H, display: "grid", placeItems: "center", opacity: 0.4, fontSize: 12 }}>train to see the curves</div>;
  }
  // x by actual step value — the thinned history is not uniformly spaced by index
  const s0 = hist[0].step;
  const s1 = hist[hist.length - 1].step;
  const x = (s: number) => PAD + ((s - s0) / Math.max(s1 - s0, 1)) * (W - 2 * PAD);
  const y = (a: number) => H - PAD - a * (H - 2 * PAD);
  const path = (k: "trainAcc" | "testAcc") =>
    hist.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p[k]).toFixed(1)}`).join(" ");
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%", background: "rgba(255,255,255,0.02)", borderRadius: 8 }}>
      <line x1={PAD} x2={W - PAD} y1={y(0.95)} y2={y(0.95)} stroke="#3a4760" strokeDasharray="2 3" opacity={0.6} />
      <path d={path("trainAcc")} fill="none" stroke="#4c8dff" strokeWidth={1.6} />
      <path d={path("testAcc")} fill="none" stroke="#ff6b6b" strokeWidth={1.6} />
    </svg>
  );
}

function Slider({ label, min, max, step, value, onChange, disabled = false }: {
  label: string; min: number; max: number; step: number; value: number;
  onChange: (v: number) => void; disabled?: boolean;
}) {
  return (
    <label style={{ display: "grid", gap: 4, fontSize: 12, opacity: disabled ? 0.55 : 0.9 }}>
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#4c8dff" }} />
    </label>
  );
}

const wrap: CSSProperties = {
  display: "grid", gap: 12, padding: 16, width: 410, maxWidth: "92vw",
  background: "rgba(10,12,20,0.78)", border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12, backdropFilter: "blur(6px)",
};
const btn: CSSProperties = {
  background: "#1b2740", color: "#e8ecf5", border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8, padding: "8px 14px", fontSize: 13, cursor: "pointer",
};
