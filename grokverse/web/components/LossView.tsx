"use client";

import type { LoadedRun } from "@/lib/types";
import type { Transition } from "@/lib/types";

const W = 520;
const H = 128;
const PAD = 34;

function logX(step: number, lx0: number, lx1: number): number {
  const v = Math.log10(Math.max(step, 1));
  return PAD + ((v - lx0) / (lx1 - lx0)) * (W - 2 * PAD);
}

function poly(steps: number[], ys: number[], lx0: number, lx1: number,
              yOf: (v: number) => number): string {
  return steps
    .map((s, i) => `${i === 0 ? "M" : "L"}${logX(s, lx0, lx1).toFixed(1)},${yOf(ys[i]).toFixed(1)}`)
    .join(" ");
}

// Module-level, not defined inside LossView's render: an inline component type
// would be recreated every frame during playback and force React to unmount
// and remount both SVG panels ~60x per second.
function Panel({ title, yOf, train, test, steps, lx0, lx1, t, curX, markerLabels }: {
  title: string; yOf: (v: number) => number; train: number[]; test: number[];
  steps: number[]; lx0: number; lx1: number; t: Transition; curX: number;
  markerLabels?: boolean;
}) {
  return (
    <svg width={W} height={H} style={{ display: "block", maxWidth: "100%" }}>
      <rect x={0} y={0} width={W} height={H} fill="rgba(255,255,255,0.02)" rx={8} />
      {t.train_saturated_step != null && (
        <line x1={logX(t.train_saturated_step, lx0, lx1)} x2={logX(t.train_saturated_step, lx0, lx1)}
          y1={PAD - 8} y2={H - PAD} stroke="#888" strokeDasharray="2 3" opacity={0.55} />
      )}
      {t.test_generalized_step != null && (
        <line x1={logX(t.test_generalized_step, lx0, lx1)} x2={logX(t.test_generalized_step, lx0, lx1)}
          y1={PAD - 8} y2={H - PAD} stroke="#c9a227" strokeDasharray="4 3" opacity={0.9} />
      )}
      {markerLabels && t.train_saturated_step != null && (
        <text x={logX(t.train_saturated_step, lx0, lx1) - 4} y={H - PAD - 4}
          fill="#9aa4b8" fontSize={9.5} textAnchor="end">memorized</text>
      )}
      {markerLabels && t.test_generalized_step != null && (
        <text x={logX(t.test_generalized_step, lx0, lx1) - 4} y={H - PAD - 14}
          fill="#c9a227" fontSize={9.5} textAnchor="end">generalized</text>
      )}
      <path d={poly(steps, train, lx0, lx1, yOf)} fill="none" stroke="#4c8dff" strokeWidth={1.6} />
      <path d={poly(steps, test, lx0, lx1, yOf)} fill="none" stroke="#ff6b6b" strokeWidth={1.6} />
      <line x1={curX} x2={curX} y1={PAD - 8} y2={H - PAD} stroke="#fff" opacity={0.85} />
      <text x={PAD} y={16} fill="#cdd6e6" fontSize={12} fontWeight={600}>{title}</text>
    </svg>
  );
}

export function LossView({ run, stepIdx }: { run: LoadedRun; stepIdx: number }) {
  const steps = run.meta.logged_steps;
  const lx0 = Math.log10(Math.max(steps[0], 1));
  const lx1 = Math.log10(Math.max(steps[steps.length - 1], 1));
  const c = run.meta.curves;
  const t = run.meta.transition;

  const yAcc = (a: number) => H - PAD - a * (H - 2 * PAD);
  const losses = [...c.train_loss, ...c.test_loss].filter((v) => v > 0);
  const lmin = Math.log10(Math.min(...losses));
  const lmax = Math.log10(Math.max(...losses));
  const yLoss = (l: number) =>
    H - PAD - ((Math.log10(Math.max(l, 1e-12)) - lmin) / (lmax - lmin)) * (H - 2 * PAD);

  const curX = logX(steps[Math.min(stepIdx, steps.length - 1)], lx0, lx1);
  const shared = { steps, lx0, lx1, t, curX };

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <Panel title="Accuracy — train (blue) vs test (red)" yOf={yAcc}
        train={c.train_acc} test={c.test_acc} markerLabels {...shared} />
      <Panel title="Loss (log scale)" yOf={yLoss}
        train={c.train_loss} test={c.test_loss} {...shared} />
    </div>
  );
}
