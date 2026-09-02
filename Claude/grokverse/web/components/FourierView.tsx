"use client";

import type { LoadedRun } from "@/lib/types";

const W = 520;
const H = 130;
const PAD = 30;

export function FourierView({ run, stepIdx }: { run: LoadedRun; stepIdx: number }) {
  const { freqs, fraction } = run.meta.fourier_spectrum;
  const dom = new Set(run.meta.dominant_frequencies);
  const maxF = Math.max(...fraction, 1e-9);
  const bw = (W - 2 * PAD) / freqs.length;

  // The bars are the FINAL spectrum (that is what the export contains); the
  // live element is the measured key-frequency concentration at the scrubbed
  // step, drawn from the per-step progress_measure — never interpolated.
  const conc = run.meta.progress_measure;
  const concNow = conc[Math.min(stepIdx, conc.length - 1)];
  const meterW = (W - 2 * PAD) * Math.max(0, Math.min(1, concNow));

  return (
    <svg width={W} height={H} style={{ display: "block", maxWidth: "100%" }} data-testid="fourier-view">
      <text x={PAD} y={16} fill="#cdd6e6" fontSize={12} fontWeight={600}>
        Final embedding Fourier spectrum — key k = {run.meta.dominant_frequencies.join(", ")}
      </text>
      <rect x={0} y={0} width={W} height={H} fill="rgba(255,255,255,0.02)" rx={8} />
      {freqs.map((k, i) => {
        const h = (fraction[i] / maxF) * (H - 2 * PAD - 20);
        const x = PAD + i * bw;
        return (
          <rect key={k} x={x} y={H - PAD - 14 - h} width={Math.max(bw - 0.5, 0.6)} height={h}
            fill={dom.has(k) ? "#ffd166" : "#4c8dff"} opacity={dom.has(k) ? 1 : 0.5} />
        );
      })}
      <text x={W - PAD} y={H - PAD - 16} fill="#7a8699" fontSize={10} textAnchor="end">frequency k →</text>
      <rect x={PAD} y={H - PAD - 2} width={W - 2 * PAD} height={5} rx={2.5} fill="rgba(255,255,255,0.06)" />
      <rect x={PAD} y={H - PAD - 2} width={meterW} height={5} rx={2.5} fill="#ffd166" opacity={0.9} />
      <text x={PAD} y={H - 8} fill="#9aa4b8" fontSize={9.5}>
        key-frequency power at this step: {(concNow * 100).toFixed(0)}%
      </text>
    </svg>
  );
}
