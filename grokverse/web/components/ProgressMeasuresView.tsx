"use client";

import { useMemo } from "react";

import type { LoadedRun } from "@/lib/types";

const W = 520;
const H = 150;
const PAD = 34;

// Series colors: gold ties "restricted" to the key frequencies highlighted in
// FourierView; violet is a new entity; gray is the recessive full-loss
// reference. Deliberately NOT LossView's blue/red, which mean train/test.
const COL_FULL = "#9aa4b8";
const COL_RESTRICTED = "#ffd166";
const COL_EXCLUDED = "#b58cff";

export function ProgressMeasuresView({ run, stepIdx }: { run: LoadedRun; stepIdx: number }) {
  const pm = run.meta.progress_measures;

  // Same log-x mapping and range as LossView so the scrub cursor lines up
  // vertically across the stacked panels.
  const loggedSteps = run.meta.logged_steps;
  const lx0 = Math.log10(Math.max(loggedSteps[0], 1));
  const lx1 = Math.log10(Math.max(loggedSteps[loggedSteps.length - 1], 1));
  const x = (step: number) =>
    PAD + ((Math.log10(Math.max(step, 1)) - lx0) / (lx1 - lx0)) * (W - 2 * PAD);

  const chance = Math.log(run.meta.p);

  // The three ~100-point path strings depend only on the run, not the scrub
  // position — memoized so 60fps playback only moves the cursor line.
  const series = useMemo(() => {
    if (!pm) return null;
    const all = [...pm.full_loss, ...pm.restricted_loss, ...pm.excluded_loss, chance]
      .filter((v) => v > 0);
    const lmin = Math.log10(Math.min(...all));
    const lmax = Math.log10(Math.max(...all));
    const y = (l: number) =>
      H - PAD - ((Math.log10(Math.max(l, 1e-12)) - lmin) / (lmax - lmin)) * (H - 2 * PAD - 14);
    const path = (ys: number[]) =>
      pm.measured_steps
        .map((s, i) => `${i === 0 ? "M" : "L"}${x(s).toFixed(1)},${y(ys[i]).toFixed(1)}`)
        .join(" ");
    return {
      yChance: y(chance),
      lines: [
        { label: "full", color: COL_FULL, d: path(pm.full_loss), width: 1.4 },
        { label: "restricted (key freqs only)", color: COL_RESTRICTED, d: path(pm.restricted_loss), width: 2 },
        { label: "excluded (key freqs removed)", color: COL_EXCLUDED, d: path(pm.excluded_loss), width: 2 },
      ],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  if (!pm || !series) return null;

  const curX = x(loggedSteps[Math.min(stepIdx, loggedSteps.length - 1)]);

  return (
    <svg width={W} height={H} style={{ display: "block", maxWidth: "100%" }} data-testid="progress-measures">
      <rect x={0} y={0} width={W} height={H} fill="rgba(255,255,255,0.02)" rx={8} />
      <text x={PAD} y={16} fill="#cdd6e6" fontSize={12} fontWeight={600}>
        Progress measures (Nanda 2023) — the circuit is the key frequencies
      </text>
      <line x1={PAD} x2={W - PAD} y1={series.yChance} y2={series.yChance}
        stroke="#7a8699" strokeDasharray="3 4" opacity={0.6} />
      <text x={W - PAD} y={series.yChance - 4} fill="#7a8699" fontSize={9.5} textAnchor="end">
        chance ln {run.meta.p}
      </text>
      {series.lines.map((s) => (
        <path key={s.label} d={s.d} fill="none" stroke={s.color} strokeWidth={s.width} />
      ))}
      <line x1={curX} x2={curX} y1={PAD - 8} y2={H - PAD} stroke="#fff" opacity={0.85} />
      {series.lines.map((s, i) => (
        <g key={s.label} transform={`translate(${PAD + i * 168}, ${H - 10})`}>
          <circle cx={4} cy={-3.5} r={3.5} fill={s.color} />
          <text x={12} y={0} fill="#aeb6c6" fontSize={10}>{s.label}</text>
        </g>
      ))}
    </svg>
  );
}
