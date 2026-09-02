"use client";

import type { CSSProperties } from "react";

import type { LoadedRun, RunIndexEntry } from "@/lib/types";

const sel: CSSProperties = {
  flex: 1, background: "#0c1018", color: "#e8ecf5",
  border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8,
  padding: "6px 8px", fontSize: 13,
};
const btn: CSSProperties = {
  background: "#1b2740", color: "#e8ecf5",
  border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8,
  padding: "6px 14px", fontSize: 13, cursor: "pointer", whiteSpace: "nowrap",
};

/** Human-readable run label; states acceleration and non-grokking honestly. */
function runLabel(r: RunIndexEntry): string {
  const parts = [
    r.arch === "mlp" ? "MLP" : r.arch,
    r.task, `frac ${r.train_frac}`, `wd ${r.weight_decay}`, `seed ${r.seed}`,
    r.grokfast ? "Grokfast" : "un-accelerated",
  ];
  if (r.transition.test_generalized_step == null) parts.push("did not grok");
  return parts.join(" · ");
}

export function ControlPanel({
  index, runId, onSelect, run, stepIdx, onScrub, playing, onTogglePlay,
}: {
  index: RunIndexEntry[];
  runId: string | null;
  onSelect: (id: string) => void;
  run: LoadedRun;
  stepIdx: number;
  onScrub: (i: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
}) {
  const T = run.meta.coords_shape[0];
  const i = Math.min(stepIdx, T - 1);
  const step = run.meta.logged_steps[i];
  const c = run.meta.curves;

  return (
    <div style={{
      display: "grid", gap: 9,
      background: "rgba(10,12,20,0.74)", border: "1px solid rgba(255,255,255,0.09)",
      borderRadius: 12, padding: "12px 14px", backdropFilter: "blur(6px)",
    }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <select value={runId ?? ""} onChange={(e) => onSelect(e.target.value)} style={sel}
          aria-label="Select training run" title={runId ?? undefined}>
          {index.map((r) => (
            <option key={r.id} value={r.id}>{runLabel(r)}</option>
          ))}
        </select>
        <button onClick={onTogglePlay} style={btn}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
      </div>
      <input
        type="range" min={0} max={T - 1} step={1} value={i}
        onChange={(e) => onScrub(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#4c8dff" }}
        aria-label="Training step"
      />
      <div style={{ display: "flex", gap: 18, fontSize: 12.5, opacity: 0.9, fontVariantNumeric: "tabular-nums" }}>
        <span>step <b>{step.toLocaleString()}</b></span>
        <span style={{ color: "#7cb0ff" }}>train acc {c.train_acc[i].toFixed(3)}</span>
        <span style={{ color: "#ff8b8b" }}>test acc {c.test_acc[i].toFixed(3)}</span>
        {run.meta.transition.grok_gap != null ? (
          <span style={{ color: "#c9a227" }}>grok gap {run.meta.transition.grok_gap.toLocaleString()}</span>
        ) : (
          <span style={{ color: "#8893a7" }}>did not grok</span>
        )}
      </div>
    </div>
  );
}
