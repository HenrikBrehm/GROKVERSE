"""H3 — is top-k Fourier concentration a waveform artifact? (INTERFACES §14, PREREGISTRATION §3)

WHY THIS MODULE EXISTS
----------------------
H3 is the study's **proposed primary contribution**, and it is a claim about a *metric*, not about a
model:

> Does top-k Fourier concentration misclassify a structured harmonic representation as less structured?

A square wave at fundamental ``k`` puts its power on the odd harmonics ``3k, 5k, 7k``, which at
``p = 113`` alias to scattered indices. A metric that keeps the top-8 **individual** frequencies
therefore scores a perfectly clean square-wave circuit as unstructured, while scoring an equally clean
sinusoidal one as highly structured. That is the mechanism this module measures.

THE CRITERION THAT IS FORBIDDEN, AND WHY IT IS ASSERTED HERE
-------------------------------------------------------------
"The harmonic family beats the cardinality-matched top-m concentration" is **unsatisfiable by
construction**: top-m is the argmax over frequency sets of size m, so ``family_fraction <=
matched_top_m_fraction`` always. It must never be used as support for H3. Every output of this module
carries ``family_minus_matched_top_m`` (which is ``<= 0``) together with that statement, and
``tests/test_h3_validity.py`` asserts the inequality rather than trusting it. The statistic that
carries the inferential load is the odd-versus-even harmonic **shape**.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It never compares the two architectures. H3₀ ("the architecture gap in structured-neuron fraction is
the same under the top-1 and the family definitions") and its refutation criteria are *paired across
seeds*, so they belong to the aggregation stage. This module produces the per-run quantities that
stage consumes, per checkpoint, per architecture, and nothing more.

DEPENDENCE ON THE EVIDENCE GATE (verbatim, master prompt §12)
--------------------------------------------------------------
    H3 does not presuppose that either architecture uses a Fourier circuit. Each architecture is first
    tested independently for periodic internal structure, the predicted phase relationship, end-to-end
    logit fit, and causal relevance. Harmonic-family concentration may be interpreted as the same
    Fourier principle in a different form only for an architecture that passes these tests.

Numbers from this module are therefore *not interpretable as H3 evidence* until `decision_tree`
reports that the architecture passed the gate. The output repeats that in its own ``params``.

STATUS: measurement code only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import Config
from . import key_frequencies as KF
from . import metrics as M
from .common import envelope, load_model_at, summarize as dist_summary, write_result
from .fourier import harmonic_family_concentration
from .mlp_mechanism import (CURVE_NAMES, PRIMARY_DEFINITION, PRIMARY_KEY_RULE,
                            activation_analysis, curve_spectra, effective_curves, neuron_tables,
                            resolve_key_frequencies, structured_neuron_definitions)

MODULE = "h3_validity"
MODULE_VERSION = "1.0"
#: Seeded random draws for the random-m null (INTERFACES §0).
N_CONTROL = 50
#: Cardinalities swept for the legacy H3c number instead of inventing one choice.
LEGACY_N_F = (1, 2, 4, 8)
#: The two structured-neuron definitions H3b contrasts (PREREGISTRATION §4.2).
FAMILY_DEFINITION = PRIMARY_DEFINITION          # "primary_family_0.50"
TOP1_DEFINITION = "top1_0.50"
#: Concentration cardinalities reported for every population.
TOPK = (1, 4, 8)


# --------------------------------------------------------------------------- #
# H3a — synthetic populations at matched (k, phase, amplitude)                 #
# --------------------------------------------------------------------------- #
def curve_parameters(Y: np.ndarray, p: int) -> dict:
    """Each column's dominant frequency, phase and amplitude, in the §0 phase convention.

    The amplitude is calibrated against a unit-amplitude reference rather than assumed from the
    basis normalization, so it stays correct if `fourier_basis` ever changes its scaling.
    """
    spec = curve_spectra(np.asarray(Y, dtype=np.float64), p)
    n = np.arange(p)
    ref = np.cos(2 * np.pi * 1 * n / p)[:, None]
    ref_power = float(curve_spectra(ref, p)["power"].max())
    amp = np.sqrt(np.maximum(spec["power"].max(axis=0), 0.0) / ref_power) if ref_power > 0 else \
        np.zeros(spec["power"].shape[1])
    return {"k": np.asarray(spec["dominant_freq"], dtype=np.int64),
            "phase": np.asarray(spec["dominant_phase"], dtype=np.float64),
            "amplitude": np.asarray(amp, dtype=np.float64),
            "dominant_fraction": np.asarray(spec["dominant_fraction"], dtype=np.float64)}


def synthetic_populations(params: dict, p: int) -> dict[str, np.ndarray]:
    """Two populations with **identical** ``(k, φ, A)``: pure sinusoids and discrete square waves.

    ``sinusoid[:, i] = A_i cos(w_{k_i} n − φ_i)`` and
    ``square[:, i]   = A_i sign(cos(w_{k_i} n − φ_i))``, the sign convention of
    `metrics.discrete_square_reference` (``cos >= 0 -> +1``).

    Because the two differ **only** in waveform, any difference in a concentration metric between them
    is a property of the metric, not of the data — which is exactly what H3a measures.
    """
    n = np.arange(p)[:, None]
    k = np.asarray(params["k"], dtype=np.float64)[None, :]
    phi = np.asarray(params["phase"], dtype=np.float64)[None, :]
    amp = np.asarray(params["amplitude"], dtype=np.float64)[None, :]
    arg = 2 * np.pi * k * n / p - phi
    return {"sinusoid": amp * np.cos(arg),
            "square": amp * np.where(np.cos(arg) >= 0, 1.0, -1.0)}


def _population_metrics(Y: np.ndarray, p: int) -> dict:
    """Concentration, family share and harmonic shape of a curve population, summarized."""
    met = M.curve_metrics(np.asarray(Y, dtype=np.float64), p, TOPK)
    shares = met["harmonic_shares"]
    out = {f"top{k}_concentration": dist_summary(met["topk_concentration"][k]) for k in TOPK}
    out.update({
        "family_share": dist_summary(shares["family_share"]),
        "fundamental_share": dist_summary(shares["fundamental_share"]),
        "odd_share": dist_summary(shares["odd_share"]),
        "even_share": dist_summary(shares["even_share"]),
        "odd_minus_even": dist_summary(shares["odd_minus_even"]),
        "spectral_entropy": dist_summary(met["spectral_entropy"]),
        "n_collisions": dist_summary(np.asarray(shares["n_collisions"], dtype=np.float64)),
    })
    return out


def waveform_sensitivity(curves: dict, p: int) -> dict:
    """H3a: what the metric does to the same circuit written as a sinusoid and as a square wave."""
    per_curve: dict = {}
    for name, Y in curves.items():
        params = curve_parameters(Y, p)
        pops = synthetic_populations(params, p)
        block = {kind: _population_metrics(pop, p) for kind, pop in pops.items()}
        sens = {f"top{k}": (block["sinusoid"][f"top{k}_concentration"]["median"]
                            - block["square"][f"top{k}_concentration"]["median"]) for k in TOPK}
        block["waveform_sensitivity_median"] = sens
        block["shape_separation_median"] = (
            block["square"]["odd_minus_even"]["median"]
            - block["sinusoid"]["odd_minus_even"]["median"])
        block["source_parameters"] = {
            "n_curves": int(params["k"].size),
            "dominant_frequency_histogram": {int(kk): int(c) for kk, c in
                                             zip(*np.unique(params["k"], return_counts=True))},
            "amplitude": dist_summary(params["amplitude"]),
        }
        per_curve[name] = block
    return {
        "per_curve": per_curve,
        "definition": ("two populations built from THIS checkpoint's own dominant (k, phase, "
                       "amplitude) per neuron, differing only in waveform; "
                       "waveform_sensitivity = median top-k(sinusoid) - median top-k(square)"),
        "reading": ("a positive waveform_sensitivity means the concentration metric scores the square "
                    "population as less concentrated although both populations carry the same "
                    "fundamentals — a property of the METRIC, not of the trained model"),
    }


# --------------------------------------------------------------------------- #
# H3b — harmonic-aware structure, per neuron, with both mandatory controls     #
# --------------------------------------------------------------------------- #
def _random_set_null(q: np.ndarray, m: int, n_control: int, seed: int) -> np.ndarray:
    """Share held by ``m`` random frequencies, drawn independently per column, ``n_control`` times.

    Independent draws per column are the right null for a *per-neuron* statistic: a single shared
    random set would measure the spectrum of the population, not of each curve.
    """
    half, n = q.shape
    if m <= 0 or m > half:
        return np.zeros((0, n))
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(n_control), n))
    for d in range(int(n_control)):
        idx = np.argsort(rng.random((half, n)), axis=0)[:m]          # m distinct rows per column
        out[d] = np.take_along_axis(q, idx, axis=0).sum(axis=0)
    return out


def harmonic_aware_structure(curves: dict, p: int, n_control: int, seed: int) -> dict:
    """H3b: the family fraction against the top-1 fraction, each with its two controls."""
    per_curve: dict = {}
    arrays: dict = {}
    for name, Y in curves.items():
        Y = np.asarray(Y, dtype=np.float64)
        spec = M.power_spectrum(Y, p)
        q = M.normalized_power(spec["power"])                        # [half, n], columns sum to 1
        k = np.asarray(M.dominant_frequency(spec["power"]), dtype=np.int64)
        shares = M.harmonic_shares(spec["power"], k, p)
        family = np.asarray(shares["family_share"], dtype=np.float64)
        top1 = np.asarray(M.topk_concentration(spec["power"], 1), dtype=np.float64)
        # `metrics.harmonic_shares` defines family_share = fundamental + ODD harmonics, i.e. the
        # frequencies {k, 3k, 5k, 7k} — cardinality 4 at max_harmonic=7, NOT the 7 that
        # `harmonic_indices` returns (it also carries the even harmonics, which the family excludes).
        # The matched control must sit at the family's own cardinality, which is the top-4 that
        # PREREGISTRATION §3 H3b names. Collisions shrink the effective family and are reported per
        # curve rather than corrected for silently.
        m = len([j for j in range(1, M.MAX_ODD_HARMONIC + 1) if j % 2 == 1])
        matched = np.asarray(M.topk_concentration(spec["power"], m), dtype=np.float64)
        null = _random_set_null(q, m, n_control, seed)
        null_mean = null.mean(axis=0) if null.size else np.zeros_like(family)
        per_curve[name] = {
            "family_fraction": dist_summary(family),
            "top1_fraction": dist_summary(top1),
            "matched_top_m_fraction": dist_summary(matched),
            "matched_m": m,
            "random_m_null": {"n": int(null.shape[0]) if null.size else 0,
                              "mean": dist_summary(null_mean),
                              "per_curve_mean_over_draws": dist_summary(null_mean)},
            "family_minus_matched_top_m": dist_summary(family - matched),
            "family_minus_random_null": dist_summary(family - null_mean),
            "odd_minus_even": dist_summary(np.asarray(shares["odd_minus_even"], dtype=np.float64)),
            "n_collisions": dist_summary(np.asarray(shares["n_collisions"], dtype=np.float64)),
            "max_family_minus_matched_top_m": float((family - matched).max()),
        }
        arrays[f"h3b__{name}__family_fraction"] = family
        arrays[f"h3b__{name}__top1_fraction"] = top1
        arrays[f"h3b__{name}__matched_top_m_fraction"] = matched
        arrays[f"h3b__{name}__random_null_mean"] = null_mean
        arrays[f"h3b__{name}__dominant_frequency"] = k
    return {"per_curve": per_curve,
            "controls": ("cardinality-matched top-m of the SAME curve (m = |family|) and a random-m "
                         "null drawn independently per curve, both mandatory (master prompt §12)"),
            "forbidden_criterion": (
                "'the family beats the matched top-m' is unsatisfiable by construction "
                "(top-m is the argmax over size-m sets), so family_minus_matched_top_m <= 0 always. "
                "It is reported as an implementation check and must NEVER be used as support for H3; "
                "the discriminating statistic is the odd-versus-even harmonic shape.")}, arrays


def structured_fractions(curves: dict, p: int, key_freqs) -> dict:
    """The structured-neuron fraction under the family and the top-1 definition (H3b's pair).

    These two per-run numbers are what the aggregation stage pairs across seeds to test H3₀. This
    module deliberately stops here: an architecture comparison needs both architectures and all seeds.
    """
    tables = neuron_tables(curves, p)
    act = activation_analysis(curves, p, key_freqs, n_exemplars=0)
    defs = structured_neuron_definitions(tables, act)
    out = {"n_live": int(defs["n_live"]), "definitions": {}}
    for name in (FAMILY_DEFINITION, TOP1_DEFINITION):
        if name in defs["definitions"]:
            entry = defs["definitions"][name]
            out["definitions"][name] = {k: v for k, v in entry.items() if k != "mask"}
    out["note"] = ("H3_0 is the statement that the ARCHITECTURE GAP in these fractions is the same "
                   "under both definitions; that comparison is paired across seeds and is not made "
                   "here")
    return out, {f"h3b__mask__{n}": defs["definitions"][n]["mask"]
                 for n in (FAMILY_DEFINITION, TOP1_DEFINITION) if n in defs["definitions"]}


# --------------------------------------------------------------------------- #
# H3c — the legacy embedding number, descriptive only                          #
# --------------------------------------------------------------------------- #
def legacy_embedding_concentration(state: dict, cfg: Config, n_control: int, seed: int) -> dict:
    """H3c: top-8 and harmonic-family concentration of the embedding object, with both controls."""
    emb, source = KF.embedding_matrix(state, cfg)
    out: dict = {"object": source, "by_n_fundamentals": {}}
    for n_f in LEGACY_N_F:
        try:
            block = harmonic_family_concentration(emb, cfg.p, n_f, n_null=int(n_control),
                                                  seed=int(seed))
        except ValueError as exc:                       # zero-power embedding: recorded, not hidden
            out["by_n_fundamentals"][str(n_f)] = {"error": str(exc)}
            continue
        out["by_n_fundamentals"][str(n_f)] = block
    # `emb` is [p, d]: each column is a function of the token index, which is the orientation
    # `power_spectrum` expects. Summing the per-column power gives the AGGREGATE embedding spectrum
    # — the object the legacy top-8 number was computed on.
    spec = M.power_spectrum(np.asarray(emb, dtype=np.float64), cfg.p)
    out["top8_concentration_of_the_object"] = float(
        np.asarray(M.topk_concentration(spec["power"].sum(axis=1, keepdims=True), 8)).ravel()[0])
    out["caveat"] = ("W_E alone decides nothing for the MLP, which reads it through two halves of "
                     "W_in; this number is the LEGACY object and is reported descriptively only "
                     "(PREREGISTRATION §3, H3c)")
    return out


# --------------------------------------------------------------------------- #
# run-level analysis                                                           #
# --------------------------------------------------------------------------- #
def _curves_for(state: dict, cfg: Config) -> dict[str, np.ndarray]:
    """The effective curves, from whichever mechanism module owns this architecture.

    Returns ``u_a``, ``u_b``, ``out`` **and** ``b_in``: the three named curves feed the per-curve
    spectral analyses, while ``b_in`` (1-D, one entry per neuron) is needed by
    `activation_analysis` and must therefore not be iterated over as if it were a curve.
    """
    if cfg.arch == "transformer":
        from . import transformer_mechanism as TM
        attn = TM.attention_weights(state, cfg)
        abar = attn.reshape(cfg.p * cfg.p, cfg.n_heads, 3).mean(axis=0)
        return TM.effective_curves(state, cfg, abar)
    return effective_curves(state, cfg)


def analyse(run_dir, step: int | None = None, key_rule: str = PRIMARY_KEY_RULE, seed: int = 0,
            n_control: int = N_CONTROL) -> tuple[dict, Path]:
    """Run H3a, H3b and H3c on one checkpoint and write the JSON + npz result."""
    run_dir = Path(run_dir)
    cfg, _model, state, meta = load_model_at(run_dir, step)
    st = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    p = cfg.p
    key_freqs, key_info = resolve_key_frequencies(run_dir, step, key_rule, state, cfg)
    curves = _curves_for(st, cfg)
    spectral = {name: curves[name] for name in CURVE_NAMES}     # the [p, n] curves only

    h3a = waveform_sensitivity(spectral, p)
    h3b, h3b_arrays = harmonic_aware_structure(spectral, p, int(n_control), int(seed))
    fractions, mask_arrays = structured_fractions(curves, p, key_freqs)
    h3c = legacy_embedding_concentration(st, cfg, int(n_control), int(seed))

    params = {
        "step": meta["step"], "seed": int(seed), "n_control": int(n_control),
        "topk": list(TOPK), "legacy_n_fundamentals": list(LEGACY_N_F),
        "max_harmonic": int(M.MAX_ODD_HARMONIC),
        "family_definition": FAMILY_DEFINITION, "top1_definition": TOP1_DEFINITION,
        "key_frequency_selection": key_info,
        "key_frequencies": [int(k) for k in key_freqs],
        "aliasing_caveat": (f"at p={p} the odd-harmonic families of different fundamentals can alias "
                            "onto one another, so 'the harmonic model fits at k' does not identify k "
                            "as the fundamental (tests/test_wave_fitting.py); n_collisions is "
                            "reported per curve and bounds what H3b can claim"),
        "gate_dependence": ("H3 does not presuppose that either architecture uses a Fourier circuit. "
                            "These numbers may be read as H3 evidence only for an architecture that "
                            "passes the evidence gate (PREREGISTRATION §5, master prompt §12)."),
        "no_architecture_comparison_here": ("H3_0 and its refutation criteria are paired across "
                                            "seeds and belong to the aggregation stage"),
        "status": "MEASUREMENT ONLY — thresholds pending docs/HUMAN_DECISIONS.md B1/B7",
    }
    payload = envelope(MODULE, MODULE_VERSION, meta, params)
    payload["results"] = {
        "h3a_waveform_sensitivity": h3a,
        "h3b_harmonic_aware_structure": h3b,
        "h3b_structured_fractions": fractions,
        "h3c_legacy_embedding": h3c,
    }
    arrays = dict(h3b_arrays)
    arrays.update(mask_arrays)
    path = write_result(run_dir, MODULE, meta["tag"], payload, arrays)
    return payload, path


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 metric-validity analysis (INTERFACES §14)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--key-rule", default=PRIMARY_KEY_RULE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--all-checkpoints", action="store_true")
    args = ap.parse_args()

    steps: list[int | None] = [args.step]
    if args.all_checkpoints:
        ck = Path(args.run_dir) / "checkpoints.json"
        if not ck.exists():
            raise SystemExit(f"{args.run_dir}: no checkpoints.json (legacy run?)")
        steps = [int(e["step"]) for e in json.loads(ck.read_text())]
    for s in steps:
        payload, path = analyse(args.run_dir, s, args.key_rule, args.seed, args.n_control)
        ua = payload["results"]["h3a_waveform_sensitivity"]["per_curve"]["u_a"]
        print(f"{payload['run_id']} step={payload['step']}  "
              f"waveform_sensitivity(top8)={ua['waveform_sensitivity_median']['top8']:+.4f}  "
              f"shape_separation={ua['shape_separation_median']:+.4f}  -> {path}")


if __name__ == "__main__":
    main()
