"use client";

import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

import { loadIndex, loadRun } from "@/lib/data";
import type { LoadedRun, RunIndexEntry } from "@/lib/types";

import { ControlPanel } from "./ControlPanel";
import { EmbeddingView3D } from "./EmbeddingView3D";
import { FourierView } from "./FourierView";
import { GuidedTour, type TourControls, type TourPhase } from "./GuidedTour";
import { LiveLab } from "./LiveLab";
import { LossView } from "./LossView";
import { ProgressMeasuresView } from "./ProgressMeasuresView";

/** Map a narrative phase to a logged-step index using the run's own measured
 *  curves — never a hard-coded fraction, so the tour cannot contradict the data. */
function phaseIndex(run: LoadedRun, phase: TourPhase): number {
  const T = run.meta.coords_shape[0];
  const { train_acc } = run.meta.curves;
  const conc = run.meta.progress_measure;
  if (phase === "start") return 0;
  if (phase === "memorized") {
    const i = train_acc.findIndex((a) => a >= 0.99);
    return i >= 0 ? i : Math.floor((T - 1) / 3);
  }
  if (phase === "structureForming") {
    // first frame where the key-frequency concentration has covered 30% of its
    // total rise — the embedding is visibly mid-reorganization
    const c0 = conc[0];
    const c1 = conc[conc.length - 1];
    const i = conc.findIndex((c) => c >= c0 + 0.3 * (c1 - c0));
    return i > 0 ? i : Math.floor((T - 1) * 0.7);
  }
  return T - 1; // "end"
}

export function Explorer() {
  const [index, setIndex] = useState<RunIndexEntry[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<LoadedRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stepF, setStepF] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [showLab, setShowLab] = useState(false);
  const [tour, setTour] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  const runRef = useRef<LoadedRun | null>(null);
  useEffect(() => { runRef.current = run; }, [run]);

  // A tour seek requested while the target run is still loading is applied
  // once the load resolves (instead of being clobbered by the reset-to-0).
  const pendingSeek = useRef<TourPhase | null>(null);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    if (mq.matches) setPlaying(false);
    const onChange = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    loadIndex()
      .then((ix) => {
        if (!ix.length) throw new Error("index.json lists no runs");
        setIndex(ix);
        setRunId(ix[0].id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!runId) return;
    let stale = false; // last-selected run wins, not last-resolved fetch
    const ctrl = new AbortController();
    loadRun(runId, ctrl.signal)
      .then((r) => {
        if (stale) return;
        setRun(r);
        setError(null);
        const seek = pendingSeek.current;
        pendingSeek.current = null;
        setStepF(seek ? phaseIndex(r, seek) : 0);
      })
      .catch((e) => { if (!stale) setError(String(e)); });
    return () => { stale = true; ctrl.abort(); };
  }, [runId]);

  const stepFRef = useRef(0);
  useEffect(() => { stepFRef.current = stepF; }, [stepF]);

  const raf = useRef<number | null>(null);
  useEffect(() => {
    if (!run || !playing || showLab) return;
    const T = run.meta.coords_shape[0];
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const next = Math.min(T - 1, stepFRef.current + dt * 12);
      setStepF(next);
      if (next >= T - 1) { setPlaying(false); return; } // stop the loop at the end
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [run, playing, showLab]);

  const tourControls: TourControls = useMemo(() => ({
    selectRun: (id) => setRunId(id),
    seekPhase: (phase) => {
      setPlaying(false);
      const r = runRef.current;
      if (r) setStepF(phaseIndex(r, phase));
      pendingSeek.current = phase; // re-applied if a run switch is in flight
    },
    setPlaying: (p) => setPlaying(p),
    showLab: (b) => setShowLab(b),
  }), []);

  const tourEntries = useMemo(() => ({
    txf: index.find((r) => r.arch === "transformer"),
    mlp: index.find((r) => r.arch === "mlp"),
  }), [index]);

  if (error) {
    return (
      <Centered>
        Could not load run data.
        <span style={{ opacity: 0.7, fontSize: 13 }}>{error}</span>
        <span style={{ opacity: 0.7, fontSize: 13 }}>
          Export first: <code>python -m grokverse.export --all</code>
        </span>
      </Centered>
    );
  }
  if (!run) return <Centered>Loading…</Centered>;

  const T = run.meta.coords_shape[0];
  const stepIdx = Math.round(Math.min(stepF, T - 1));
  const grokked = run.meta.transition.test_generalized_step != null;

  return (
    <main style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden" }}>
      {/* Scope notice — the explorer serves the reproduction-phase runs; the pre-registered
          architecture study is reported in RESULTS.md and not yet reflected here (E3). */}
      <div data-testid="study-notice" style={{
        position: "absolute", top: 8, left: "50%", transform: "translateX(-50%)", zIndex: 5,
        maxWidth: "min(760px, 92vw)", padding: "6px 12px", borderRadius: 8, fontSize: 12, lineHeight: 1.35,
        color: "#e8ecf5", background: "rgba(10,12,20,0.72)", border: "1px solid rgba(255,255,255,0.12)",
        backdropFilter: "blur(6px)", textAlign: "center",
      }}>
        This explorer shows the <strong>reproduction-phase runs (June 2026)</strong>. The pre-registered
        architecture study (September 2026, 10 paired seeds, evidence gate <code>neither_passes</code>) is
        reported in <code>RESULTS.md</code>; this view has not been updated to it (see{" "}
        <code>docs/dev/EXPLORER_UPDATE_PLAN.md</code>).
      </div>
      <Canvas camera={{ position: [0, 0, 3.4], fov: 50 }} dpr={[1, 2]} style={{ position: "absolute", inset: 0 }}>
        <EmbeddingView3D run={run} stepF={stepF} autoRotate={!reducedMotion} />
      </Canvas>

      <header style={{ position: "absolute", top: 26, left: 30, pointerEvents: "none", maxWidth: 360 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, letterSpacing: "0.04em" }}>GROKVERSE</h1>
        <p style={{ opacity: 0.62, fontSize: 13, marginTop: 4, lineHeight: 1.5 }}>
          {grokked
            ? `${run.meta.p} token embeddings reorganizing into periodic structure as the ${run.meta.arch === "mlp" ? "MLP" : "network"} groks.`
            : `${run.meta.p} token embeddings of a run that memorized but never grokked — the structure does not emerge.`}
        </p>
      </header>

      <div style={{ position: "absolute", top: 26, right: 24, display: "flex", gap: 8 }}>
        <button style={hdrBtn} onClick={() => setTour(true)} data-testid="start-tour">✦ Guided tour</button>
        <button style={hdrBtn} onClick={() => setShowLab((s) => !s)} data-testid="toggle-lab">
          {showLab ? "← Explorer" : "⚗ Live Lab"}
        </button>
      </div>

      {showLab ? (
        <div style={{ position: "absolute", top: 82, right: 24 }} data-testid="livelab">
          <LiveLab />
        </div>
      ) : (
        <>
          <div style={{ position: "absolute", top: 82, right: 24, display: "grid", gap: 10, width: 540, maxWidth: "44vw" }}>
            <Glass><LossView run={run} stepIdx={stepIdx} /></Glass>
            {run.meta.progress_measures && (
              <Glass><ProgressMeasuresView run={run} stepIdx={stepIdx} /></Glass>
            )}
            <Glass><FourierView run={run} stepIdx={stepIdx} /></Glass>
          </div>
          <div style={{ position: "absolute", bottom: 24, left: "50%", transform: "translateX(-50%)", width: 560, maxWidth: "92vw" }}>
            <ControlPanel
              index={index} runId={runId}
              onSelect={(id) => { pendingSeek.current = null; setRunId(id); }} run={run}
              stepIdx={stepIdx} onScrub={(i) => { setPlaying(false); setStepF(i); }}
              playing={playing}
              onTogglePlay={() => {
                if (!playing && stepF >= T - 1) setStepF(0); // Play at the end restarts
                setPlaying((p) => !p);
              }}
            />
          </div>
        </>
      )}

      {tour && (
        <GuidedTour
          controls={tourControls} entries={tourEntries}
          onClose={() => { pendingSeek.current = null; setTour(false); }}
        />
      )}
    </main>
  );
}

function Centered({ children }: { children: ReactNode }) {
  return (
    <div style={{
      display: "grid", placeItems: "center", gap: 8, textAlign: "center",
      width: "100vw", height: "100vh", padding: 24,
    }}>
      {children}
    </div>
  );
}

function Glass({ children }: { children: ReactNode }) {
  return (
    <div style={{
      background: "rgba(10,12,20,0.66)", border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 12, padding: 10, backdropFilter: "blur(6px)",
    }}>
      {children}
    </div>
  );
}

const hdrBtn: CSSProperties = {
  background: "rgba(27,39,64,0.8)", color: "#e8ecf5", border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8, padding: "7px 13px", fontSize: 13, cursor: "pointer", backdropFilter: "blur(6px)",
};
