// Types mirroring the Python export data contract
// (training/grokverse/export.py -> web/public/data/*).

export interface Transition {
  train_saturated_step: number | null;
  test_generalized_step: number | null;
  grok_gap: number | null;
  final_train_acc: number | null;
  final_test_acc: number | null;
}

export interface Curves {
  train_loss: number[];
  test_loss: number[];
  train_acc: number[];
  test_acc: number[];
}

export interface FourierSpectrum {
  freqs: number[];
  power: number[];
  fraction: number[];
  const_power: number;
  total_power: number;
}

export interface RunIndexEntry {
  id: string;
  arch: string;
  task: string;
  p: number;
  weight_decay: number;
  train_frac: number;
  seed: number;
  grokfast: boolean;
  dominant_frequencies: number[];
  transition: Transition;
}

/** Nanda 2023 restricted/excluded-loss curves; only present for runs where
 *  analysis/progress_measures.py was actually computed — never synthesized. */
export interface ProgressMeasures {
  measured_steps: number[];
  full_loss: number[];
  restricted_loss: number[];
  excluded_loss: number[];
  key_frequencies: number[];
}

export interface RunMeta {
  id: string;
  arch: string;
  task: string;
  p: number;
  hyperparams: Record<string, number>;
  seed: number;
  lib_versions: Record<string, string>;
  git_commit: string;
  logged_steps: number[];
  curves: Curves;
  progress_measure: number[];
  dominant_frequencies: number[];
  fourier_spectrum: FourierSpectrum;
  transition: Transition;
  coords_shape: [number, number, number]; // [T, p, 3]
  explained_variance_ratio: number[];
  progress_measures?: ProgressMeasures;
}

export interface LoadedRun {
  meta: RunMeta;
  coords: Float32Array; // length T*p*3, row-major [T, p, 3]
}
