"use client";

import type { CSSProperties } from "react";
import { useEffect, useState } from "react";

import type { RunIndexEntry } from "@/lib/types";

/** Narrative positions resolved against the loaded run's own measured curves
 *  (see Explorer.phaseIndex) — the tour never hard-codes a step fraction. */
export type TourPhase = "start" | "memorized" | "structureForming" | "end";

export type TourControls = {
  selectRun: (id: string) => void;
  seekPhase: (phase: TourPhase) => void;
  setPlaying: (p: boolean) => void;
  showLab: (b: boolean) => void;
};

export type TourEntries = { txf?: RunIndexEntry; mlp?: RunIndexEntry };

type Step = {
  title: string;
  body: (e: TourEntries) => string;
  lab?: boolean;
  run: (c: TourControls, e: TourEntries) => void;
};

const fmt = (n: number) => n.toLocaleString("en-US");

/** Honest phrasing for the architecture-comparison step, built from the two
 *  index entries actually shown. Only quotes a ratio when the two runs share
 *  the same training condition (frac + acceleration); otherwise it names the
 *  conditions and defers to RESULTS.md for the matched comparison. */
function mlpBody(e: TourEntries): string {
  const { txf, mlp } = e;
  if (!mlp) return "No MLP run is exported yet.";
  const g = mlp.transition.test_generalized_step;
  if (g == null) return "This MLP run memorized but never grokked in its step budget.";
  const cond = `${mlp.grokfast ? "Grokfast-accelerated" : "un-accelerated"}, train frac ${mlp.train_frac}`;
  const tg = txf?.transition.test_generalized_step;
  const matched = txf && tg != null &&
    txf.grokfast === mlp.grokfast && txf.train_frac === mlp.train_frac;
  if (matched) {
    const ratio = (g / tg).toFixed(1);
    return `A 2-layer MLP groks the same task too — generalizing at step ${fmt(g)} (${cond}) ` +
      `vs the transformer's ${fmt(tg)}: ~${ratio}× slower under identical conditions, ` +
      `and with a less sparse embedding. Same phenomenon, different inductive bias.`;
  }
  return `A 2-layer MLP groks the same task too — this run generalizes at step ${fmt(g)} ` +
    `(${cond}, so not directly comparable to the previous run's timing). Under matched ` +
    `conditions the transformer groks ~3× faster with a sparser embedding — see RESULTS.md §3.`;
}

const STEPS: Step[] = [
  {
    title: "Welcome to GROKVERSE",
    body: () => "A 1-layer transformer learning (a + b) mod 113. Each point is one number's learned embedding.",
    run: (c, e) => { if (e.txf) c.selectRun(e.txf.id); c.setPlaying(false); c.seekPhase("start"); },
  },
  {
    title: "1 · Memorization",
    body: (e) => {
      const t = e.txf?.transition.train_saturated_step;
      return `${t != null ? `By step ${fmt(t)} the` : "The"} network has MEMORIZED the training set ` +
        "(watch the train accuracy below) — yet the embedding is still a featureless blob, and test " +
        "accuracy sits at chance. It has understood nothing.";
    },
    run: (c) => c.seekPhase("memorized"),
  },
  {
    title: "2 · The moment it 'gets it'",
    body: () => "Scrub forward. Long after memorizing, the points suddenly begin to reorganize…",
    run: (c) => c.seekPhase("structureForming"),
  },
  {
    title: "3 · Structure emerges",
    body: () => "…into a periodic RING. The network discovered the circular structure of modular arithmetic. This is grokking.",
    run: (c) => c.seekPhase("end"),
  },
  {
    title: "4 · The phase transition",
    body: (e) => {
      const g = e.txf?.transition.test_generalized_step;
      const gap = e.txf?.transition.grok_gap;
      return `The loss panel marks the exact step where test accuracy jumps (gold dashed line` +
        `${g != null ? `, step ${fmt(g)}` : ""}) — generalization, ` +
        `${gap != null ? `${fmt(gap)} steps` : "long"} after train accuracy saturated.`;
    },
    run: (c) => c.seekPhase("end"),
  },
  {
    title: "5 · The circuit",
    body: () => "The Fourier panel shows the few key frequencies the embedding now uses — a sparse trig-identity circuit, not a memorized lookup table.",
    run: (c) => c.seekPhase("end"),
  },
  {
    title: "6 · A different architecture",
    body: mlpBody,
    run: (c, e) => { if (e.mlp) c.selectRun(e.mlp.id); c.seekPhase("end"); },
  },
  {
    title: "7 · Your turn",
    body: () => "Open the Live Lab and train a network in your own browser. Raise the weight decay and watch grokking appear.",
    lab: true,
    run: () => {},
  },
];

export function GuidedTour({
  controls, entries, onClose,
}: {
  controls: TourControls;
  entries: TourEntries;
  onClose: () => void;
}) {
  const [i, setI] = useState(0);
  useEffect(() => {
    controls.showLab(Boolean(STEPS[i].lab)); // Back out of the lab step closes it
    STEPS[i].run(controls, entries);
  }, [i, controls, entries]);

  const step = STEPS[i];
  const last = i === STEPS.length - 1;

  return (
    <div style={card} data-testid="guided-tour">
      <div style={{ fontSize: 11, letterSpacing: "0.08em", opacity: 0.5, textTransform: "uppercase" }}>
        Guided tour · {i + 1}/{STEPS.length}
      </div>
      <h3 style={{ fontSize: 17, fontWeight: 700 }}>{step.title}</h3>
      <p style={{ fontSize: 13.5, lineHeight: 1.55, opacity: 0.85 }}>{step.body(entries)}</p>
      <div style={{ display: "flex", gap: 8, justifyContent: "space-between", alignItems: "center" }}>
        <button style={ghost} onClick={onClose} data-testid="tour-skip">Skip</button>
        <div style={{ display: "flex", gap: 8 }}>
          <button style={ghost} disabled={i === 0} onClick={() => setI((v) => Math.max(0, v - 1))}>Back</button>
          <button style={primary} data-testid="tour-next" onClick={() => (last ? onClose() : setI((v) => v + 1))}>
            {last ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}

const card: CSSProperties = {
  position: "absolute", bottom: 110, left: 30, width: 340, maxWidth: "90vw",
  display: "grid", gap: 8, padding: "16px 18px", zIndex: 20,
  background: "rgba(12,15,24,0.92)", border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 14, backdropFilter: "blur(8px)",
};
const primary: CSSProperties = {
  background: "#2f6df6", color: "#fff", border: "none", borderRadius: 8,
  padding: "7px 16px", fontSize: 13, cursor: "pointer", fontWeight: 600,
};
const ghost: CSSProperties = {
  background: "transparent", color: "#aeb6c6", border: "1px solid rgba(255,255,255,0.14)",
  borderRadius: 8, padding: "7px 12px", fontSize: 13, cursor: "pointer",
};
