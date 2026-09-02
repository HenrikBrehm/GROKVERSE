"""Checks for the two mechanism derivations (run: python tests/test_derivations.py).

Pins every algebraic claim of docs/MLP_MECHANISM_DERIVATION.md and
docs/TRANSFORMER_MECHANISM_DERIVATION.md on RANDOM models — no trained run is read, so this
file can run before, during and after the study without touching any result.

Claims checked
--------------
transformer   the forward decomposition (embedding -> attention row 2 -> residual -> ReLU MLP ->
              unembedding) reproduces model.logits_last; logits = direct path + MLP path; the
              per-neuron contributions through W_out @ W_U sum to the MLP path; the mean-attention
              effective curves are EXACT when attention is uniform (W_Q = W_K = 0)
mlp           the effective curves u_a, u_b, b_in, W_out, b_out reproduce the forward pass
ReLU algebra  rectifying cos(w a - phi_a) + cos(w b - phi_b) produces an (a+b) term whose phase is
              phi_a + phi_b and whose amplitude is 8/(3 pi^2)
square waves  a DISCRETE square wave at odd p: fundamental share, odd-harmonic share 1/9+1/25+1/49,
              small but nonzero even-harmonic share; aliased odd harmonics of one fundamental are
              distinct at prime p
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grokverse.config import get_config  # noqa: E402
from grokverse.data import make_dataset  # noqa: E402
from grokverse.models import build_model  # noqa: E402
from grokverse.seed import set_seed  # noqa: E402

CROSS_TERM_AMPLITUDE = 8.0 / (3.0 * math.pi ** 2)      # 0.27019, derivation §4
IDEAL_SQUARE_ODD_SHARE = 1 / 9 + 1 / 25 + 1 / 49       # 0.17151, derivation §5
IDEAL_SQUARE_FUNDAMENTAL_SHARE = 8.0 / math.pi ** 2    # 0.81057


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def _small_transformer(seed: int = 0):
    cfg = get_config("nanda", p=23, d_model=32, n_heads=2, d_head=16, d_mlp=24, seed=seed)
    set_seed(seed)
    model = build_model(cfg)
    model.eval()
    return cfg, model


def _decompose(cfg, model, x: torch.Tensor) -> dict:
    """The forward decomposition of TRANSFORMER_MECHANISM_DERIVATION §2, in float64."""
    W_E, W_pos = model.W_E.detach().double(), model.W_pos.detach().double()
    W_Q, W_K = model.W_Q.detach().double(), model.W_K.detach().double()
    W_V, W_O = model.W_V.detach().double(), model.W_O.detach().double()
    W_in, W_out = model.W_in.detach().double(), model.W_out.detach().double()
    W_U = model.W_U.detach().double()

    emb = W_E[x] + W_pos[None, :cfg.n_ctx, :]
    q = torch.einsum("btd,hde->bhte", emb, W_Q)
    k = torch.einsum("btd,hde->bhte", emb, W_K)
    scores = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(cfg.d_head)
    causal = torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool), diagonal=1)
    attn_row = scores.masked_fill(causal, float("-inf")).softmax(-1)[:, :, cfg.n_ctx - 1, :]
    v = torch.einsum("btd,hde->bhte", emb, W_V)
    z = torch.einsum("bhs,bhse->bhe", attn_row, v)
    r_pre = emb[:, cfg.n_ctx - 1, :] + torch.einsum("bhe,hed->bd", z, W_O)
    hidden = torch.relu(r_pre @ W_in)
    neuron_logit_map = W_out @ W_U
    return {
        "emb": emb, "attn_row": attn_row, "r_pre": r_pre, "hidden": hidden,
        "neuron_logit_map": neuron_logit_map,
        "direct": r_pre @ W_U, "mlp": hidden @ neuron_logit_map,
        "logits": r_pre @ W_U + hidden @ neuron_logit_map,
        "OV": torch.einsum("hde,hef->hdf", W_V, W_O), "W_in": W_in,
        "W_E": W_E, "W_pos": W_pos,
    }


def _effective_curves(cfg, dec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-attention effective operand curves, derivation §4."""
    abar = dec["attn_row"].mean(0)                                  # [h, n_ctx]
    W_E, W_pos, OV, W_in = dec["W_E"], dec["W_pos"], dec["OV"], dec["W_in"]
    tok_a = W_E[:cfg.p] + W_pos[0]
    tok_b = W_E[:cfg.p] + W_pos[1]
    tok_eq = W_E[cfg.equals_token] + W_pos[cfg.n_ctx - 1]
    u_a = torch.einsum("h,pd,hdf,fm->pm", abar[:, 0], tok_a, OV, W_in)
    u_b = torch.einsum("h,pd,hdf,fm->pm", abar[:, 1], tok_b, OV, W_in)
    const = (tok_eq @ W_in) + torch.einsum(
        "h,d,hdf,fm->m", abar[:, cfg.n_ctx - 1], tok_eq, OV, W_in)
    return u_a, u_b, const


def check_transformer_decomposition():
    print("\n-- transformer forward decomposition (TRANSFORMER_MECHANISM_DERIVATION §2, §3) --")
    cfg, model = _small_transformer()
    x = make_dataset(cfg)["all_x"]
    with torch.no_grad():
        want = model.logits_last(x).double()
    dec = _decompose(cfg, model, x)

    check("decomposition reproduces model.logits_last (<1e-5)",
          float((dec["logits"] - want).abs().max()) < 1e-5)
    check("logits split exactly into direct path + MLP path",
          float((dec["direct"] + dec["mlp"] - want).abs().max()) < 1e-5)
    contrib = dec["hidden"][:, :, None] * dec["neuron_logit_map"][None, :, :]
    check("per-neuron contributions sum to the MLP path (<1e-12)",
          float((contrib.sum(1) - dec["mlp"]).abs().max()) < 1e-12)
    check("neuron_logit_map has shape [d_mlp, vocab]",
          tuple(dec["neuron_logit_map"].shape) == (cfg.d_mlp, cfg.vocab_size))
    check("attention row is a distribution over the 3 positions",
          float((dec["attn_row"].sum(-1) - 1).abs().max()) < 1e-9)


def check_effective_curves_transformer():
    print("\n-- transformer effective curves (derivation §4) --")
    cfg, model = _small_transformer(seed=1)
    x = make_dataset(cfg)["all_x"]

    # (a) uniform attention => the additive reconstruction is EXACT
    with torch.no_grad():
        model.W_Q.zero_()
        model.W_K.zero_()
    dec = _decompose(cfg, model, x)
    check("W_Q = W_K = 0 gives a uniform causal attention row",
          float((dec["attn_row"] - 1.0 / cfg.n_ctx).abs().max()) < 1e-6)
    u_a, u_b, const = _effective_curves(cfg, dec)
    approx = u_a[x[:, 0]] + u_b[x[:, 1]] + const
    true_pre = dec["r_pre"] @ dec["W_in"]
    check("uniform attention: effective curves reproduce the pre-activation (<1e-10)",
          float((approx - true_pre).abs().max()) < 1e-10)

    # (b) additivity_r2 is a measurement in [0, 1], = 1 in the uniform case
    ss_res = ((true_pre - approx) ** 2).sum(0)
    ss_tot = ((true_pre - true_pre.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    check("uniform attention: additivity_r2 == 1 for every neuron",
          float(r2.min()) > 1 - 1e-9)

    # (c) with input-dependent attention it is below 1 but still well defined
    cfg2, model2 = _small_transformer(seed=2)
    with torch.no_grad():                       # amplify Q/K so attention really varies
        model2.W_Q.mul_(25.0)
        model2.W_K.mul_(25.0)
    dec2 = _decompose(cfg2, model2, x)
    ua2, ub2, c2 = _effective_curves(cfg2, dec2)
    ap2 = ua2[x[:, 0]] + ub2[x[:, 1]] + c2
    tp2 = dec2["r_pre"] @ dec2["W_in"]
    r2b = 1 - ((tp2 - ap2) ** 2).sum(0) / ((tp2 - tp2.mean(0)) ** 2).sum(0)
    spread = float(dec2["attn_row"][:, :, 0].std())
    check("strong Q/K makes attention input-dependent (std > 0.01)", spread > 0.01)
    check("input-dependent attention: additivity_r2 drops below 1",
          float(r2b.min()) < 1 - 1e-6)
    check("additivity_r2 stays a finite number <= 1",
          bool(torch.isfinite(r2b).all()) and float(r2b.max()) <= 1 + 1e-9)


def check_effective_curves_mlp():
    print("\n-- MLP effective curves (MLP_MECHANISM_DERIVATION §2) --")
    cfg = get_config("nanda", p=23, arch="mlp", d_model=32, d_mlp=24, seed=0)
    set_seed(0)
    model = build_model(cfg)
    model.eval()
    st = {k: v.detach().double() for k, v in model.state_dict().items()}
    d = cfg.d_model
    u_a = st["W_E"] @ st["W_in"][:d]
    u_b = st["W_E"] @ st["W_in"][d:]
    x = make_dataset(cfg)["all_x"]
    with torch.no_grad():
        want = model.logits_last(x).double()
    act = torch.relu(u_a[x[:, 0]] + u_b[x[:, 1]] + st["b_in"])
    got = act @ st["W_out"] + st["b_out"]
    check("effective curves reproduce the MLP forward pass (<1e-5)",
          float((got - want).abs().max()) < 1e-5)
    check("u_a / u_b have shape [p, d_mlp]",
          tuple(u_a.shape) == (cfg.p, cfg.d_mlp) and tuple(u_b.shape) == (cfg.p, cfg.d_mlp))
    check("the MLP pre-activation is exactly additive in (a, b)",
          float((u_a[x[:, 0]] + u_b[x[:, 1]] + st["b_in"]
                 - (torch.cat([st["W_E"][x[:, 0]], st["W_E"][x[:, 1]]], -1) @ st["W_in"]
                    + st["b_in"])).abs().max()) < 1e-9)


def check_relu_cross_term():
    print("\n-- ReLU produces the (a+b) cross term with phase phi_a + phi_b (§4) --")
    p, k = 23, 3
    w = 2 * np.pi * k / p
    n = np.arange(p)
    A, B = np.meshgrid(n, n, indexing="ij")
    for phi_a, phi_b in ((0.7, -1.3), (0.0, 0.0), (2.5, 2.5)):
        r = np.maximum(np.cos(w * A - phi_a) + np.cos(w * B - phi_b), 0.0)
        c_amp = (r * np.cos(w * (A + B))).sum() * 2 / p ** 2
        s_amp = (r * np.sin(w * (A + B))).sum() * 2 / p ** 2
        phase = np.arctan2(s_amp, c_amp)
        err = abs(np.angle(np.exp(1j * (phase - (phi_a + phi_b)))))
        amp = float(np.hypot(c_amp, s_amp))
        check(f"phase(cross term) == phi_a + phi_b for ({phi_a}, {phi_b}) (err {err:.1e})",
              err < 5e-3)
        check(f"cross-term amplitude ~ 8/(3 pi^2) for ({phi_a}, {phi_b}) (got {amp:.4f})",
              abs(amp - CROSS_TERM_AMPLITUDE) < 0.02)

    # §4.1: the (a-b) term is EXACTLY as large, with phase phi_a - phi_b. A single neuron
    # does not prefer addition; asserting otherwise would pre-register a false prediction.
    for phi_a, phi_b in ((0.7, -1.3), (1.1, 0.3)):
        r = np.maximum(np.cos(w * A - phi_a) + np.cos(w * B - phi_b), 0.0)
        s_c = (r * np.cos(w * (A + B))).sum() * 2 / p ** 2
        s_s = (r * np.sin(w * (A + B))).sum() * 2 / p ** 2
        d_c = (r * np.cos(w * (A - B))).sum() * 2 / p ** 2
        d_s = (r * np.sin(w * (A - B))).sum() * 2 / p ** 2
        s_amp, d_amp = float(np.hypot(s_c, s_s)), float(np.hypot(d_c, d_s))
        check(f"single neuron: (a-b) amplitude equals (a+b) amplitude ({d_amp:.4f} vs {s_amp:.4f})",
              abs(d_amp - s_amp) < 0.01)
        d_err = abs(np.angle(np.exp(1j * (np.arctan2(d_s, d_c) - (phi_a - phi_b)))))
        check(f"single neuron: (a-b) phase equals phi_a - phi_b (err {d_err:.1e})", d_err < 5e-3)

    # a purely linear unit produces NO cross term at all (the ReLU is the multiplier)
    lin = np.cos(w * A - 0.7) + np.cos(w * B + 1.3)
    lin_amp = np.hypot((lin * np.cos(w * (A + B))).sum(),
                       (lin * np.sin(w * (A + B))).sum()) * 2 / p ** 2
    check("without the ReLU there is no (a+b) term", lin_amp < 1e-10)


def check_population_selects_sum():
    """§4.1: the readout, not the neuron, selects (a+b) — the (a-b) content cancels."""
    print("\n-- the population + readout select (a+b) (§4.1) --")
    p, k, n_neurons = 23, 3, 400
    w = 2 * np.pi * k / p
    n = np.arange(p)
    A, B = np.meshgrid(n, n, indexing="ij")
    rng = np.random.default_rng(0)
    phi_a = rng.uniform(-np.pi, np.pi, n_neurons)
    phi_b = rng.uniform(-np.pi, np.pi, n_neurons)

    def population(out_phase) -> np.ndarray:
        logits = np.zeros((p, p, p))
        for i in range(n_neurons):
            act = np.maximum(np.cos(w * A - phi_a[i]) + np.cos(w * B - phi_b[i]), 0.0)
            logits += act[:, :, None] * np.cos(w * n - out_phase[i])[None, None, :]
        return logits - logits.mean(-1, keepdims=True)

    S = np.cos(w * (A[:, :, None] + B[:, :, None] - n[None, None, :]))
    D = np.cos(w * (A[:, :, None] - B[:, :, None] - n[None, None, :]))

    def share(logits, M) -> float:
        coef = (logits * M).sum() / (M * M).sum()
        return float(1 - ((logits - coef * M) ** 2).sum() / (logits ** 2).sum())

    lg = population(phi_a + phi_b)                       # the H1 readout
    r2_sum, r2_diff = share(lg, S), share(lg, D)
    check(f"H1 readout: logits are mostly cos(w(a+b-c)) (R2 {r2_sum:.3f})", r2_sum > 0.8)
    check(f"H1 readout: the (a-b) content cancels (R2 {r2_diff:.3f})", r2_diff < 0.05)

    # negative control: a scrambled readout phase destroys the (a+b) structure
    bad = population(rng.uniform(-np.pi, np.pi, n_neurons))
    check(f"scrambled readout: no (a+b) structure (R2 {share(bad, S):.3f})", share(bad, S) < 0.2)

    # and the difference readout selects (a-b) instead -- the mechanism is symmetric
    dif = population(phi_a - phi_b)
    check(f"difference readout selects (a-b) instead (R2 {share(dif, D):.3f})", share(dif, D) > 0.8)


def check_square_wave_harmonics():
    print("\n-- discrete square waves and aliasing (§5) --")
    p = 113
    n = np.arange(p)

    def alias(j, k):
        r = (j * k) % p
        return min(r, p - r)

    for k in (1, 5, 18, 56):
        sq = np.sign(np.cos(2 * np.pi * k * n / p + 1e-9))
        power = np.abs(np.fft.rfft(sq)) ** 2
        total = power[1:].sum()
        fund = power[alias(1, k)]
        odd = sum(power[alias(j, k)] for j in (3, 5, 7))
        even = sum(power[alias(j, k)] for j in (2, 4, 6))
        check(f"k={k}: fundamental share ~ 8/pi^2 (got {fund/total:.4f})",
              abs(fund / total - IDEAL_SQUARE_FUNDAMENTAL_SHARE) < 0.01)
        check(f"k={k}: odd-harmonic share ~ 1/9+1/25+1/49 (got {odd/fund:.4f})",
              abs(odd / fund - IDEAL_SQUARE_ODD_SHARE) < 0.01)
        check(f"k={k}: even-harmonic share is small but nonzero (got {even/fund:.2e})",
              0 < even / fund < 0.01)
        check(f"k={k}: aliased odd harmonics are distinct from each other and from k",
              len({alias(j, k) for j in (1, 3, 5, 7)}) == 4)

    # a sinusoid has no harmonic power at all -- the negative control
    sine = np.cos(2 * np.pi * 5 * n / p)
    pw = np.abs(np.fft.rfft(sine)) ** 2
    check("a sinusoid puts no power at its odd harmonics",
          sum(pw[alias(j, 5)] for j in (3, 5, 7)) / pw[5] < 1e-20)

    # collisions DO occur across a set of fundamentals -- the reason the statistic is per curve
    legacy = [18, 15, 11, 1, 13, 22, 56, 36]
    collisions = [(j, k, alias(j, k)) for k in legacy for j in (2, 3, 4, 5, 6, 7)
                  if alias(j, k) in legacy]
    check("the legacy top-8 set has harmonic collisions (>= 5)", len(collisions) >= 5)


def main():
    check_transformer_decomposition()
    check_effective_curves_transformer()
    check_effective_curves_mlp()
    check_relu_cross_term()
    check_population_selects_sum()
    check_square_wave_harmonics()
    print("\nALL DERIVATION CHECKS PASSED")


if __name__ == "__main__":
    main()
