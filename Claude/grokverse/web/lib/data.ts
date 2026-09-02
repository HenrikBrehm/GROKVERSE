import type { LoadedRun, RunIndexEntry, RunMeta } from "./types";

const BASE = "/data";

export async function loadIndex(): Promise<RunIndexEntry[]> {
  const res = await fetch(`${BASE}/index.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`index.json ${res.status}`);
  return res.json();
}

export async function loadRun(id: string, signal?: AbortSignal): Promise<LoadedRun> {
  const [metaRes, binRes] = await Promise.all([
    fetch(`${BASE}/${id}.meta.json`, { cache: "no-store", signal }),
    fetch(`${BASE}/${id}.coords.bin`, { cache: "no-store", signal }),
  ]);
  if (!metaRes.ok) throw new Error(`${id}.meta.json ${metaRes.status}`);
  if (!binRes.ok) throw new Error(`${id}.coords.bin ${binRes.status}`);
  const meta: RunMeta = await metaRes.json();
  const coords = new Float32Array(await binRes.arrayBuffer());

  // Contract checks: a truncated blob or a meta/coords mismatch must fail
  // loudly here, not render silently-wrong geometry (PROMPT.md §6).
  const [T, p, k] = meta.coords_shape;
  if (T < 2) throw new Error(`${id}: coords_shape[0]=${T}, need >= 2 logged steps to render`);
  if (coords.length !== T * p * k) {
    throw new Error(`${id}.coords.bin holds ${coords.length} floats, expected ${T * p * k} from coords_shape`);
  }
  if (meta.logged_steps.length !== T || meta.curves.test_acc.length !== T ||
      meta.progress_measure.length !== T) {
    throw new Error(`${id}: logged_steps/curves/progress_measure length does not match coords_shape[0]=${T}`);
  }
  return { meta, coords };
}

/** Linearly interpolated token coordinates [p*3] at a fractional step index. */
export function lerpCoords(run: LoadedRun, f: number): Float32Array {
  const [T, p] = run.meta.coords_shape;
  const stride = p * 3;
  const clamped = Math.max(0, Math.min(T - 1, f));
  const i0 = Math.floor(clamped);
  const i1 = Math.min(T - 1, i0 + 1);
  const t = clamped - i0;
  const a = run.coords.subarray(i0 * stride, (i0 + 1) * stride);
  const b = run.coords.subarray(i1 * stride, (i1 + 1) * stride);
  const out = new Float32Array(stride);
  for (let k = 0; k < stride; k++) out[k] = a[k] * (1 - t) + b[k] * t;
  return out;
}
