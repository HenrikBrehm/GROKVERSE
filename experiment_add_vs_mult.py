#!/usr/bin/env python3
"""experiment_add_vs_mult.py — Modulare Addition vs. Multiplikation Grokking-Vergleich.

Vergleicht systematisch das Grokking-Verhalten eines 1-Layer Transformers auf:
  Task A: c = (a + b) mod p     (additive Gruppe Z_p)
  Task B: c = (a * b) mod p     (multiplikative Gruppe F_p^x, exkl. 0)

Misst: Memorization Step, Generalization Step, Grok Gap,
       Top-8 Fourier Frequency Power (natürliche UND log-permutierte Ordnung),
       sowie zwei cap-freie Sparsity-Metriken (normalisierte Spektral-Entropie,
       Participation Ratio).

Verwendung (aus training/):
    python experiment_add_vs_mult.py --seeds 3 --steps 8000
    python experiment_add_vs_mult.py --seeds 5 --steps 15000 --no-grokfast  # Un-accelerated
    python experiment_add_vs_mult.py --self-test                            # nur Basis-Checks

Korrektur-Historie (Audit-Fixes gegenüber der Version des Kollegen):
  Bug 1 — top8_power_log_permuted() baute für gerades q = p - 1 eine
          überbestimmte, nicht-orthogonale Fourier-Basis (114 statt 112
          Zeilen für q=112; die Nyquist-Frequenz wurde doppelt eingebaut,
          einmal über eine entartete, durch Rauschen normierte Sinus-Zeile).
          Fix: siehe real_dft_basis() weiter unten.
  Bug 2 — top8_power() schnitt für 'mul' die nie-trainierte Zeile W_E[0]
          mit ein und injizierte damit Rauschen ins Spektrum. Fix: siehe
          Docstring von top8_power().
  Bug 3 — Nur Top-8 (cap-behaftet) und Default 3 Seeds, obwohl genau das
          im Audit kritisiert wurde. Fix: Default 5 Seeds + zusätzliche
          cap-freie Metriken (Entropie, Participation Ratio).
  Bug 4 — log_schedule() quantisiert den erkannten Grokking-Übergang auf
          bis zu ~150 Schritte, ohne das zu kennzeichnen. Fix: siehe
          find_crossing() weiter unten — gibt immer ein Intervall zurück.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. Konfiguration
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExpConfig:
    p: int = 113
    task: str = "add"           # "add" oder "mul"
    seed: int = 0
    train_frac: float = 0.5
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    steps: int = 8000
    grokfast: bool = True
    grokfast_alpha: float = 0.98
    grokfast_lambda: float = 2.0
    n_logged_steps: int = 200

    @property
    def run_id(self) -> str:
        gf = f"_gf{self.grokfast_lambda}" if self.grokfast else ""
        return f"txf_{self.task}_p{self.p}_wd{self.weight_decay}_frac{self.train_frac}{gf}_seed{self.seed}"


# ═══════════════════════════════════════════════════════════════
# 2. Seeding
# ═══════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# 3. Dataset: Addition vs. Multiplikation
# ═══════════════════════════════════════════════════════════════

def make_dataset(cfg: ExpConfig) -> dict:
    p = cfg.p

    if cfg.task == "add":
        # Alle (a, b) Paare, a, b in [0, p)
        a = torch.arange(p).repeat_interleave(p)
        b = torch.arange(p).repeat(p)
        c = (a + b) % p
    elif cfg.task == "mul":
        # Nur (a, b) mit a, b in [1, p) — multiplikative Gruppe F_p^x
        vals = torch.arange(1, p)
        a = vals.repeat_interleave(p - 1)
        b = vals.repeat(p - 1)
        c = (a * b) % p
    else:
        raise ValueError(f"Unbekannter Task: {cfg.task}")

    eq_token = p  # '=' Token
    eq = torch.full_like(a, eq_token)
    x = torch.stack([a, b, eq], dim=1).long()
    y = c.long()

    n_total = x.shape[0]
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=g)
    n_train = int(round(cfg.train_frac * n_total))
    return {
        "train_x": x[perm[:n_train]], "train_y": y[perm[:n_train]],
        "test_x": x[perm[n_train:]], "test_y": y[perm[n_train:]],
        "all_x": x, "all_y": y,
    }


# ═══════════════════════════════════════════════════════════════
# 4. Modell: 1-Layer Transformer (identisch zu Grokverse)
# ═══════════════════════════════════════════════════════════════

class OneLayerTransformer(nn.Module):
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        d, v = cfg.d_model, cfg.p + 1
        h, dh = cfg.n_heads, cfg.d_head
        assert h * dh == d

        self.W_E = nn.Parameter(torch.empty(v, d))
        self.W_pos = nn.Parameter(torch.empty(3, d))
        self.W_Q = nn.Parameter(torch.empty(h, d, dh))
        self.W_K = nn.Parameter(torch.empty(h, d, dh))
        self.W_V = nn.Parameter(torch.empty(h, d, dh))
        self.W_O = nn.Parameter(torch.empty(h, dh, d))
        self.W_in = nn.Parameter(torch.empty(d, cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, d))
        self.W_U = nn.Parameter(torch.empty(d, v))
        self._init(d, dh, h, cfg.d_mlp)

    def _init(self, d, dh, h, d_mlp):
        s = 1.0
        nn.init.normal_(self.W_E, std=s / math.sqrt(d))
        nn.init.normal_(self.W_pos, std=s / math.sqrt(d))
        for W in (self.W_Q, self.W_K, self.W_V):
            nn.init.normal_(W, std=s / math.sqrt(d))
        nn.init.normal_(self.W_O, std=s / math.sqrt(dh * h))
        nn.init.normal_(self.W_in, std=s / math.sqrt(d))
        nn.init.normal_(self.W_out, std=s / math.sqrt(d_mlp))
        nn.init.normal_(self.W_U, std=s / math.sqrt(d))

    def forward(self, tokens):
        T = tokens.shape[1]
        x = self.W_E[tokens] + self.W_pos[None, :T, :]
        q = torch.einsum("btd,hde->bhte", x, self.W_Q)
        k = torch.einsum("btd,hde->bhte", x, self.W_K)
        v = torch.einsum("btd,hde->bhte", x, self.W_V)
        scores = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(self.W_Q.shape[-1])
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        z = torch.einsum("bhts,bhse->bhte", attn, v)
        x = x + torch.einsum("bhte,hed->btd", z, self.W_O)
        h = torch.relu(torch.einsum("btd,df->btf", x, self.W_in))
        x = x + torch.einsum("btf,fd->btd", h, self.W_out)
        return torch.einsum("btd,dv->btv", x, self.W_U)

    def logits_last(self, tokens):
        return self.forward(tokens)[:, -1, :]


# ═══════════════════════════════════════════════════════════════
# 5. Fourier-Analyse (natürliche UND log-permutierte Ordnung)
# ═══════════════════════════════════════════════════════════════
#
# WICHTIG (Bug 1 Fix): Eine orthonormale reelle DFT-Basis über Z_n hat
# für UNGERADES n genau n Zeilen: 1 Konstante + (n-1)/2 (cos, sin)-Paare.
# Für GERADES n gibt es zusätzlich eine Nyquist-Frequenz k = n/2, für die
# sin(2*pi*(n/2)*m/n) = sin(pi*m) für ganzzahliges m IDENTISCH NULL ist —
# es gibt dort also keinen Sinus-Partner, nur eine einzelne Kosinus-Zeile
# cos(pi*m). Zeilenanzahl für gerades n: 1 + 2*(n/2 - 1) + 1 = n.
#
# Die Original-Implementierung von top8_power_log_permuted() hat für
# q = p - 1 = 112 (gerade) fälschlich k bis EINSCHLIESSLICH q/2 = 56 in
# der (cos, sin)-Schleife laufen lassen UND danach nochmal separat eine
# Nyquist-Zeile angehängt. Für k=56 gilt aber bereits
# cos(2*pi*56*m/112) == cos(pi*m)  (== die separat angehängte nyq-Zeile,
# exakte Dopplung) und sin(2*pi*56*m/112) == sin(pi*m) == 0 bis auf
# Floating-Point-Rauschen (~1e-13) — diese Rauschzeile wurde dann per
# Division durch ihre eigene (Rauschen-)Norm zu einem Einheitsvektor
# aufgeblasen, der reale Signal-Power absorbiert. Ergebnis: 114 Zeilen
# in einem 112-dimensionalen Raum (überbestimmt), max|F F^T - I| = 1.0,
# und Energie wird nicht erhalten (Parseval verletzt, +2% in Tests).
#
# real_dft_basis() unten ist die korrekte, generische Konstruktion für
# beliebiges n (gerade ODER ungerade) und wird sowohl für die natürliche
# als auch die log-permutierte Analyse verwendet.

def real_dft_basis(n: int) -> np.ndarray:
    """Orthonormale reelle DFT-Basis über Z_n für beliebiges n >= 1.

    Zeilen: [konstant, cos_1, sin_1, cos_2, sin_2, ..., (Nyquist falls n gerade)].
    Für ungerades n: 1 + (n-1) Zeilen = n (keine Nyquist-Frequenz).
    Für gerades n:   1 + 2*(n/2 - 1) + 1 Zeilen = n (Nyquist nur Kosinus,
                     da die zugehörige Sinus-Komponente exakt 0 ist).
    In beiden Fällen also exakt n Zeilen -> quadratische, orthonormale
    Basis von R^n (siehe --self-test für den numerischen Nachweis).
    """
    idx = np.arange(n)
    rows = [np.ones(n) / np.sqrt(n)]  # k = 0 (Gleichanteil / DC)
    half = n // 2
    top_k = half if n % 2 == 1 else half - 1  # gerade n: Nyquist separat unten
    for k in range(1, top_k + 1):
        c = np.cos(2 * np.pi * k * idx / n)
        s = np.sin(2 * np.pi * k * idx / n)
        rows.append(c / np.linalg.norm(c))
        rows.append(s / np.linalg.norm(s))
    if n % 2 == 0:
        nyq = np.cos(np.pi * idx)  # k = n/2, sin-Partner waere identisch 0
        rows.append(nyq / np.linalg.norm(nyq))
    return np.stack(rows)


def frequency_band_powers(power: np.ndarray, n: int) -> np.ndarray:
    """Aggregiert Zeilen-Power (aus real_dft_basis(n)) zu Frequenzband-Power.

    power[0] ist die DC/Gleichanteil-Zeile und wird ausgeschlossen (wie im
    Original). Für k = 1 .. floor(n/2) (ohne Nyquist) werden Kosinus- und
    Sinus-Power addiert. Bei geradem n hat das letzte Band (Nyquist,
    k = n/2) NUR eine Kosinus-Komponente — das wird hier korrekt beachtet,
    statt (wie im Original) eine zweite, degenerierte Nyquist-Zeile zu
    erwarten.

    Anzahl Bänder: floor(n/2) für beide Paritäten von n. Für n=112 (q=p-1,
    p=113) sind das 55 (cos,sin)-Bänder (k=1..55) + 1 Nyquist-Band (k=56,
    nur cos) = 56 Bänder insgesamt.
    """
    half = n // 2
    top_k = half if n % 2 == 1 else half - 1
    bands = []
    for k in range(1, top_k + 1):
        cos_idx = 1 + 2 * (k - 1)
        sin_idx = 2 + 2 * (k - 1)
        bands.append(power[cos_idx] + power[sin_idx])
    if n % 2 == 0:
        nyq_idx = 1 + 2 * top_k  # letzte Zeile von real_dft_basis(n)
        bands.append(power[nyq_idx])
    return np.array(bands)


def fourier_basis(p: int) -> np.ndarray:
    """Orthonormale reelle Fourier-Basis über Z_p (Kompatibilitäts-Wrapper).

    Für ungerades p (z.B. p=113, das prime Modulus des Projekts) ist die
    Original-Implementierung bereits korrekt: es gibt keine Nyquist-
    Frequenz, jede der (p-1)/2 Frequenzen hat ein volles (cos, sin)-Paar,
    macht zusammen mit der Konstanten genau p Zeilen. Verifiziert über
    --self-test (Orthonormalität + Parseval).
    """
    return real_dft_basis(p)


def spectral_sparsity_metrics(freq_power: np.ndarray, top_n: int = 8) -> dict:
    """Drei Sparsity-Metriken auf derselben Frequenzband-Power-Verteilung.

    - top_n_power: Anteil der `top_n` stärksten Bänder an der Gesamt-Power
      (cap-behaftet, Original-Metrik — behalten für Vergleichbarkeit mit
      bestehenden Projekt-Ergebnissen).
    - entropy_norm: normalisierte Spektral-Entropie H_norm = H / log(n_bands),
      H = -sum p_k log(p_k) über die normierte Power-Verteilung p_k. Cap-frei;
      0 = maximal sparsam (ein Band dominiert), 1 = maximal breitbandig
      (Gleichverteilung über alle Bänder).
    - participation_ratio: PR = (sum p_k)^2 / sum p_k^2 auf den RAW
      Band-Powers (nicht normiert). Cap-frei; misst effektiv, wie viele
      Bänder "beteiligt" sind (PR=1: ein Band trägt alles, PR=n_bands:
      Gleichverteilung).
    """
    n_bands = len(freq_power)
    total = float(freq_power.sum())
    if total <= 0 or n_bands == 0:
        return {"top8": 0.0, "entropy_norm": 0.0, "participation_ratio": 0.0}

    sorted_power = np.sort(freq_power)[::-1]
    top_n_power = float(sorted_power[:top_n].sum() / total)

    p_k = freq_power / total
    p_k_pos = p_k[p_k > 0]
    entropy = float(-np.sum(p_k_pos * np.log(p_k_pos)))
    entropy_norm = float(entropy / np.log(n_bands)) if n_bands > 1 else 0.0

    participation_ratio = float((freq_power.sum() ** 2) / (freq_power ** 2).sum())

    return {"top8": top_n_power, "entropy_norm": entropy_norm, "participation_ratio": participation_ratio}


def primitive_root(p: int) -> int:
    """Findet die kleinste Primitivwurzel modulo p."""
    for g in range(2, p):
        seen = set()
        val = 1
        for _ in range(p - 1):
            val = (val * g) % p
            seen.add(val)
        if len(seen) == p - 1:
            return g
    raise ValueError(f"Keine Primitivwurzel für p={p} gefunden")


def discrete_log_permutation(p: int) -> np.ndarray:
    """Berechnet die Permutation [g^0, g^1, ..., g^(p-2)] mod p.
    Gibt ein Array der Länge p-1 zurück, das die Elemente 1..p-1 in
    diskreter-Logarithmus-Reihenfolge enthält."""
    g = primitive_root(p)
    perm = np.zeros(p - 1, dtype=int)
    val = 1
    for i in range(p - 1):
        perm[i] = val
        val = (val * g) % p
    return perm


def top8_power(W_E: np.ndarray, p: int, task: str = "add") -> dict:
    """Fourier-Sparsity der Token-Embeddings in natürlicher Reihenfolge.

    Bug 2 (Fix): Beim 'mul'-Task treten a, b, c ausschließlich in
    {1, ..., p-1} auf (F_p^x, die multiplikative Gruppe ohne die 0) —
    Token 0 erscheint dort NIE als Input oder Target. Unter AdamW mit
    weight_decay=1.0 zerfällt die zugehörige Zeile W_E[0] dennoch
    monoton Richtung 0 (0.999^8000 ≈ 3e-4 relative Amplitude), bleibt
    aber reines untrainiertes Rauschen statt einer gelernten
    Repräsentation. Würde man W_E[:p] (inkl. Token 0) für 'mul'
    analysieren, injiziert diese tote Zeile breitbandige Artefakt-Power
    ins Spektrum.

    Gewählte Behandlung: Für 'mul' wird Token 0 explizit ausgeschlossen;
    die Analyse läuft über die verbleibenden p-1 = 112 tatsächlich
    trainierten Zeilen W_E[1:p], in ihrer natürlichen (numerischen)
    Reihenfolge, mit der dafür korrekten 112-Punkte-DFT-Basis
    (real_dft_basis(p - 1) — derselben Konstruktion wie für die
    log-permutierte Analyse, nur ohne die Permutation). Für 'add' sind
    alle p Tokens 0..p-1 real trainiert (additive Gruppe Z_p), daher
    bleibt die Analyse dort über die volle p-Punkte-Basis wie im
    Original.
    """
    if task == "mul":
        n = p - 1
        F_mat = real_dft_basis(n)
        W = np.asarray(W_E)[1:p]  # Token 0 ausgeschlossen (nie trainiert, s.o.)
    else:
        n = p
        F_mat = real_dft_basis(n)
        W = np.asarray(W_E)[:p]

    coeffs = F_mat @ W
    power = (coeffs ** 2).sum(axis=1)
    freq_power = frequency_band_powers(power, n)
    return spectral_sparsity_metrics(freq_power)


def top8_power_log_permuted(W_E: np.ndarray, p: int) -> dict:
    """Fourier-Analyse in der diskreten-Logarithmus-permutierten Reihenfolge.
    Für Multiplikation: Sind die Embeddings in log_g-Reihenfolge periodisch?

    F_p^x ist zyklisch von Ordnung p-1 und isomorph zu Z_{p-1} via den
    diskreten Logarithmus. `perm` (aus discrete_log_permutation) enthält
    nur die Elemente 1..p-1 — Token 0 ist hier von Natur aus nie enthalten,
    das Bug-2-Problem tritt in dieser Funktion also gar nicht erst auf.
    """
    perm = discrete_log_permutation(p)
    # Extrahiere Embeddings für Elemente 1..p-1 in log_g-Reihenfolge
    W_reordered = np.asarray(W_E)[perm]  # [p-1, d]

    # Fourier-Basis über Z_{p-1} (Ordnung p-1 = 112, GERADE -> siehe
    # real_dft_basis()-Docstring für die korrekte Nyquist-Behandlung)
    q = p - 1
    F_log = real_dft_basis(q)

    coeffs = F_log @ W_reordered
    power = (coeffs ** 2).sum(axis=1)
    freq_power = frequency_band_powers(power, q)
    return spectral_sparsity_metrics(freq_power)


# ═══════════════════════════════════════════════════════════════
# 6. Training Loop
# ═══════════════════════════════════════════════════════════════

def log_schedule(steps: int, n: int) -> list[int]:
    """Geometrisch verteiltes Checkpoint-Gitter für Kurven-Logging.

    Bug 4: Nahe dem Grokking-Übergang (typischerweise step ~1000-3000 von
    8000) beträgt der Gitterabstand ~90-150 Schritte. Jeder daraus
    abgeleitete "Übergangsschritt" trägt also eine unausgesprochene
    Quantisierungsunsicherheit in dieser Größenordnung — das RESEARCH_SPEC.md
    des Projekts dokumentiert genau dieses Artefakt (verschiedene Seeds,
    die auf denselben Crossing-Step "einrasten"). Wir behalten das
    Log-Gitter für das (günstige) Kurven-Logging bei, aber `find_crossing()`
    unten gibt IMMER das umschließende Auswertungsintervall zurück, nie
    einen unmarkierten Punktschätzer.
    """
    if steps <= 1:
        return [0, 1]
    raw = np.logspace(0, np.log10(steps), n, dtype=int)
    out = sorted(set([0] + raw.tolist()))
    return out


def find_crossing(step_arr: list[int], values: list[float], threshold: float):
    """Erster Schritt, an dem `values >= threshold`, plus das umschließende
    Log-Gitter-Intervall [vorheriger geloggter Schritt, gefundener Schritt].

    Der reale Übergang liegt irgendwo in diesem Intervall — er wurde beim
    vorherigen Checkpoint noch nicht erreicht und ist beim gefundenen
    Checkpoint bereits überschritten. Rückgabe: (step, lower_bound,
    upper_bound), oder (None, None, None) falls die Schwelle nie erreicht
    wird.
    """
    for i, v in enumerate(values):
        if v >= threshold:
            lower = step_arr[i - 1] if i > 0 else step_arr[i]
            upper = step_arr[i]
            return step_arr[i], lower, upper
    return None, None, None


@torch.no_grad()
def evaluate(model, x, y, p):
    logits = model.logits_last(x)[:, :p]
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def train_run(cfg: ExpConfig) -> dict:
    set_seed(cfg.seed)
    data = make_dataset(cfg)
    train_x, train_y = data["train_x"], data["train_y"]
    test_x, test_y = data["test_x"], data["test_y"]

    model = OneLayerTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay,
                             betas=(cfg.beta1, cfg.beta2))

    logged = set(log_schedule(cfg.steps, cfg.n_logged_steps))
    curves = {k: [] for k in ("step", "train_loss", "test_loss", "train_acc", "test_acc")}

    gf_ema = {}
    for step in range(cfg.steps + 1):
        if step in logged:
            tr_l, tr_a = evaluate(model, train_x, train_y, cfg.p)
            te_l, te_a = evaluate(model, test_x, test_y, cfg.p)
            curves["step"].append(step)
            curves["train_loss"].append(tr_l)
            curves["test_loss"].append(te_l)
            curves["train_acc"].append(tr_a)
            curves["test_acc"].append(te_a)
            if step % 500 == 0 or step == cfg.steps:
                print(f"  [{cfg.run_id}] step {step:6d}  "
                      f"train[loss {tr_l:.4f} acc {tr_a:.3f}]  "
                      f"test[loss {te_l:.4f} acc {te_a:.3f}]", flush=True)

        if step == 0:
            continue

        model.train()
        logits = model.logits_last(train_x)[:, :cfg.p]
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grokfast:
            for name, prm in model.named_parameters():
                if prm.grad is None:
                    continue
                prev = gf_ema.get(name)
                ema = (prm.grad.detach().clone() if prev is None
                       else cfg.grokfast_alpha * prev
                       + (1 - cfg.grokfast_alpha) * prm.grad.detach())
                gf_ema[name] = ema
                prm.grad.add_(ema, alpha=cfg.grokfast_lambda)
        opt.step()

    final_W_E = model.W_E.detach().cpu().numpy().copy()

    # --- Transition Detection (Bug 4: immer mit Intervall, nie Punktschätzer) ---
    step_arr = curves["step"]
    train_sat, train_sat_lo, train_sat_hi = find_crossing(step_arr, curves["train_acc"], 0.99)
    test_gen, test_gen_lo, test_gen_hi = find_crossing(step_arr, curves["test_acc"], 0.95)
    gap = (test_gen - train_sat) if (train_sat is not None and test_gen is not None) else None

    # --- Fourier-Analyse (Bug 1 + Bug 2 Fixes, Bug 3: zusätzliche cap-freie Metriken) ---
    nat_metrics = top8_power(final_W_E, cfg.p, task=cfg.task)
    if cfg.task == "mul":
        log_metrics = top8_power_log_permuted(final_W_E, cfg.p)
    else:
        log_metrics = None

    return {
        "run_id": cfg.run_id,
        "task": cfg.task,
        "seed": cfg.seed,
        "train_saturated_step": train_sat,
        "train_saturated_step_interval": [train_sat_lo, train_sat_hi],
        "test_generalized_step": test_gen,
        "test_generalized_step_interval": [test_gen_lo, test_gen_hi],
        "grok_gap": gap,
        "final_train_acc": curves["train_acc"][-1],
        "final_test_acc": curves["test_acc"][-1],
        "top8_fourier_natural": nat_metrics["top8"],
        "fourier_natural_entropy_norm": nat_metrics["entropy_norm"],
        "fourier_natural_participation_ratio": nat_metrics["participation_ratio"],
        "top8_fourier_log_permuted": log_metrics["top8"] if log_metrics else None,
        "fourier_log_entropy_norm": log_metrics["entropy_norm"] if log_metrics else None,
        "fourier_log_participation_ratio": log_metrics["participation_ratio"] if log_metrics else None,
        "curves": curves,
    }


# ═══════════════════════════════════════════════════════════════
# 7. Self-Test (--self-test): schnelle Assertions, kein Training
# ═══════════════════════════════════════════════════════════════

def self_test() -> None:
    """Verifiziert die Fourier-Basen und Zahlentheorie-Hilfsfunktionen ohne
    ein Modell zu trainieren. Exit-Code 0 bei Erfolg, AssertionError sonst."""
    rng = np.random.default_rng(0)

    # --- Ungerades p (additive Gruppe Z_p, p = 113) ---
    p = 113
    F_p = fourier_basis(p)
    assert F_p.shape == (p, p), f"fourier_basis({p}) shape {F_p.shape} != ({p}, {p})"
    err_p = float(np.max(np.abs(F_p @ F_p.T - np.eye(p))))
    assert err_p < 1e-10, f"fourier_basis({p}) nicht orthonormal: max|F F^T - I| = {err_p}"
    W_p = rng.standard_normal((p, 8))
    e_in_p = float((W_p ** 2).sum())
    e_out_p = float(((F_p @ W_p) ** 2).sum())
    rel_err_p = abs(e_in_p - e_out_p) / e_in_p
    assert rel_err_p < 1e-8, f"Parseval verletzt für p={p}: in={e_in_p} out={e_out_p}"
    print(f"[self-test] fourier_basis({p}) [ungerade]: shape={F_p.shape}, "
          f"orth_err={err_p:.3e}, Parseval: {e_in_p:.4f} vs {e_out_p:.4f} (rel_err={rel_err_p:.2e})")

    # --- Gerades q = p - 1 = 112 (multiplikative Gruppe, log-permutiert & natürlich) ---
    q = p - 1
    F_q = real_dft_basis(q)
    assert F_q.shape == (q, q), f"real_dft_basis({q}) shape {F_q.shape} != ({q}, {q})"
    err_q = float(np.max(np.abs(F_q @ F_q.T - np.eye(q))))
    assert err_q < 1e-10, f"real_dft_basis({q}) nicht orthonormal: max|F F^T - I| = {err_q}"
    W_q = rng.standard_normal((q, 8))
    e_in_q = float((W_q ** 2).sum())
    e_out_q = float(((F_q @ W_q) ** 2).sum())
    rel_err_q = abs(e_in_q - e_out_q) / e_in_q
    assert rel_err_q < 1e-8, f"Parseval verletzt für q={q}: in={e_in_q} out={e_out_q}"
    print(f"[self-test] real_dft_basis({q}) [gerade]: shape={F_q.shape}, "
          f"orth_err={err_q:.3e}, Parseval: {e_in_q:.4f} vs {e_out_q:.4f} (rel_err={rel_err_q:.2e})")

    # --- Zahlentheorie ---
    g = primitive_root(113)
    assert g == 3, f"primitive_root(113) = {g}, erwartet 3"
    perm = discrete_log_permutation(113)
    assert sorted(perm.tolist()) == list(range(1, 113)), \
        "discrete_log_permutation(113) ist keine Permutation von 1..112"
    print(f"[self-test] primitive_root(113) = {g} (OK)")
    print("[self-test] discrete_log_permutation(113) ist eine gültige Permutation von 1..112 (OK)")

    print("\n[self-test] ALLE TESTS BESTANDEN")


# ═══════════════════════════════════════════════════════════════
# 7B. Analyse bestehender Runs ohne Training (--analyze-run)
# ═══════════════════════════════════════════════════════════════
#
# Wendet top8_power() / top8_power_log_permuted() (Abschnitt 5, unverändert)
# auf bereits trainierte Runs an, die unter training/runs/<run_id>/ liegen
# (embeddings.npy, shape (n_checkpoints, p+1, d_model); model_final.pt;
# run.json). Reine Nachanalyse gespeicherter Checkpoints — kein Training,
# kein Gradient. Siehe --help für --analyze-run.

def _infer_task_from_dirname(run_dir: Path) -> str | None:
    """Rät den Task ('add'/'mul') aus dem Run-Verzeichnisnamen (Fallback,
    falls run.json fehlt oder keinen Task angibt). Projekt-Konvention:
    Run-IDs enthalten '_add_' bzw. '_mul_' (siehe ExpConfig.run_id)."""
    name = run_dir.name
    if "_add_" in name:
        return "add"
    if "_mul_" in name:
        return "mul"
    return None


def _extract_p_and_task_from_run_json(meta: dict) -> tuple[int | None, str | None]:
    """Liest p/task aus einer geladenen run.json. Unterstützt sowohl das
    aktuelle Format (verschachtelt unter 'config') als auch p/task direkt
    auf oberster Ebene, falls ein run.json mal anders aufgebaut ist."""
    config = meta.get("config", meta) if isinstance(meta, dict) else {}
    p = config.get("p") if isinstance(config, dict) else None
    task = config.get("task") if isinstance(config, dict) else None
    return p, task


def load_run_for_analysis(run_dir: Path) -> dict:
    """Lädt embeddings.npy aus `run_dir` und bestimmt p/task für die
    Analyse, OHNE zu trainieren. p/task kommen bevorzugt aus run.json;
    Fallback: p = W_E.shape[0] - 1, task aus dem Verzeichnisnamen.

    Rückgabe: dict mit run_id, run_dir, p, p_source, task, task_source,
    n_checkpoints, W_E_init [p+1, d] (Checkpoint 0), W_E_final [p+1, d]
    (letzter Checkpoint, Index -1).
    """
    run_dir = Path(run_dir)
    emb_path = run_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy nicht gefunden in {run_dir}")

    embeddings = np.load(emb_path)
    if embeddings.ndim != 3:
        raise ValueError(
            f"embeddings.npy in {run_dir} hat Form {embeddings.shape}, "
            f"erwartet (n_checkpoints, p+1, d_model)"
        )

    W_E_final = embeddings[-1]
    W_E_init = embeddings[0]
    p_from_shape = int(W_E_final.shape[0] - 1)

    p, task = None, None
    p_source = "Fallback (W_E-Shape - 1)"
    task_source = "Fallback (Verzeichnisname)"

    run_json_path = run_dir / "run.json"
    if run_json_path.exists():
        try:
            meta = json.loads(run_json_path.read_text(encoding="utf-8"))
            p_json, task_json = _extract_p_and_task_from_run_json(meta)
            if p_json is not None:
                p, p_source = int(p_json), "run.json"
            if task_json is not None:
                task, task_source = str(task_json), "run.json"
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] run.json in {run_dir} konnte nicht gelesen werden ({exc}); nutze Fallback.")

    if p is not None and p != p_from_shape:
        print(f"  [warn] run.json p={p} weicht von embeddings.npy ab "
              f"(Shape impliziert p={p_from_shape}); verwende run.json-Wert.")
    if p is None:
        p = p_from_shape
    if task is None:
        task = _infer_task_from_dirname(run_dir) or "add"

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "p": p,
        "p_source": p_source,
        "task": task,
        "task_source": task_source,
        "n_checkpoints": int(embeddings.shape[0]),
        "W_E_init": W_E_init,
        "W_E_final": W_E_final,
    }


def analyze_run(run_dir: Path) -> dict:
    """Berechnet top8_power() (natürliche Ordnung) und
    top8_power_log_permuted() (diskrete-Log-Ordnung) für Init- (Checkpoint 0)
    und Final-Checkpoint (Checkpoint -1) eines bestehenden Runs.

    Die log-permutierte Metrik wird für BEIDE Tasks berechnet: für 'mul' ist
    sie die eigentliche Zielmetrik, für 'add' dient sie als klar so
    gekennzeichnete Vergleichs-Baseline (siehe Tabellen-Spalte 'ordering').
    """
    info = load_run_for_analysis(run_dir)
    p, task = info["p"], info["task"]
    g = primitive_root(p)

    rows = []
    for ckpt_label, W_E in (("init", info["W_E_init"]), ("final", info["W_E_final"])):
        nat = top8_power(W_E, p, task=task)
        log_perm = top8_power_log_permuted(W_E, p)
        for ordering, metrics in (("natural", nat), ("log_permuted", log_perm)):
            rows.append({
                "checkpoint": ckpt_label,
                "ordering": ordering,
                "top8": metrics["top8"],
                "entropy_norm": metrics["entropy_norm"],
                "participation_ratio": metrics["participation_ratio"],
            })

    return {
        "run_id": info["run_id"],
        "run_dir": info["run_dir"],
        "p": p,
        "p_source": info["p_source"],
        "task": task,
        "task_source": info["task_source"],
        "n_checkpoints": info["n_checkpoints"],
        "primitive_root": g,
        "rows": rows,
    }


def _fmt_cell(row: dict, header: str) -> str:
    key = "participation_ratio" if header == "particip_ratio" else header
    val = row.get(key, "")
    return f"{val:.4f}" if isinstance(val, float) else str(val)


def _print_analysis_table(rows: list[dict]) -> None:
    if not rows:
        print("\n  (keine Ergebnisse)")
        return
    headers = ["run_id", "checkpoint", "ordering", "top8", "entropy_norm", "particip_ratio"]
    widths = [max(len(h), max(len(_fmt_cell(r, h)) for r in rows)) for h in headers]
    print()
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(_fmt_cell(r, h).ljust(w) for h, w in zip(headers, widths)))


def _load_existing_analysis_results(out_path: Path) -> dict:
    """Lädt bereits vorhandene analyze_run_results.json (falls vorhanden und
    gültig), damit spätere --analyze-run-Aufrufe akkumulieren statt frühere
    Ergebnisse zu überschreiben (z.B. add-Runs heute, mul-Runs später)."""
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def run_analysis_mode(run_dir_args: list[str]) -> None:
    """Führt --analyze-run für jeden übergebenen Run-Ordner aus: lädt
    embeddings.npy + run.json, berechnet die Fourier-Sparsity-Metriken für
    Init- und Final-Checkpoint (natürliche + log-permutierte Ordnung),
    druckt eine kompakte Tabelle und schreibt/akkumuliert
    analyze_run_results.json neben diesem Skript. Kein Training."""
    out_path = Path(__file__).resolve().parent / "analyze_run_results.json"
    accumulated = _load_existing_analysis_results(out_path)

    table_rows = []
    for raw_dir in run_dir_args:
        run_dir = Path(raw_dir)
        print(f"\n{'=' * 70}\n  Analysiere Run: {run_dir}\n{'=' * 70}")
        try:
            result = analyze_run(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [FEHLER] {exc}")
            continue

        print(f"  p                  = {result['p']}  (Quelle: {result['p_source']})")
        print(f"  task               = {result['task']}  (Quelle: {result['task_source']})")
        print(f"  n_checkpoints      = {result['n_checkpoints']}")
        print(f"  primitive_root(p)  = {result['primitive_root']}")

        for row in result["rows"]:
            table_rows.append({"run_id": result["run_id"], **row})
        accumulated[result["run_id"]] = result

    _print_analysis_table(table_rows)

    out_path.write_text(json.dumps(accumulated, indent=2), encoding="utf-8")
    print(f"\n  [Gespeichert] {out_path}")


# ═══════════════════════════════════════════════════════════════
# 8. Hauptprogramm
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Add vs. Multiply Grokking Experiment")
    ap.add_argument("--seeds", type=int, default=5, help="Anzahl Seeds pro Task")  # Bug 3: war 3
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--no-grokfast", action="store_true", help="Ohne Grokfast-Beschleunigung")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--self-test", action="store_true",
                     help="Nur Fourier-Basen/Zahlentheorie-Assertions ausführen und beenden")
    ap.add_argument(
        "--analyze-run", action="append", default=None, metavar="RUN_DIR",
        help="Analysiere einen bestehenden Run-Ordner (embeddings.npy + run.json "
             "unter training/runs/<run_id>/) mit top8_power()/"
             "top8_power_log_permuted(), OHNE zu trainieren. Mehrfach angebbar "
             "für mehrere Runs; beendet danach das Programm (kein Training). "
             "GOVERNANCE-HINWEIS: 'txf_mul_*'-Runs sind dem menschlichen Autor "
             "vorbehalten und wurden von der KI-Implementierung dieses Flags "
             "nicht ausgeführt oder gelesen — der Owner kann sie selbst mit "
             "demselben Flag auswerten, z.B. "
             "--analyze-run training/runs/txf_mul_p113_wd1.0_frac0.5_gf2.0_seed0")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.analyze_run:
        run_analysis_mode(args.analyze_run)
        return

    tasks = ["add", "mul"]
    all_results = {}

    for task in tasks:
        task_results = []
        for seed in range(args.seeds):
            cfg = ExpConfig(
                p=args.p, task=task, seed=seed,
                train_frac=args.train_frac, steps=args.steps,
                grokfast=not args.no_grokfast,
            )
            print(f"\n{'='*60}")
            print(f"  Task: {task.upper()}  |  Seed: {seed}  |  Steps: {args.steps}")
            print(f"  Grokfast: {cfg.grokfast}  |  Run ID: {cfg.run_id}")
            print(f"{'='*60}\n")

            result = train_run(cfg)
            task_results.append(result)

            print(f"\n  => Train saturated: step {result['train_saturated_step']} "
                  f"(Intervall {result['train_saturated_step_interval']})")
            print(f"  => Test generalized: step {result['test_generalized_step']} "
                  f"(Intervall {result['test_generalized_step_interval']})")
            print(f"  => Grok gap: {result['grok_gap']}")
            print(f"  => Final test acc: {result['final_test_acc']:.4f}")
            print(f"  => Fourier (natural): top8={result['top8_fourier_natural']:.4f}  "
                  f"H_norm={result['fourier_natural_entropy_norm']:.4f}  "
                  f"PR={result['fourier_natural_participation_ratio']:.2f}")
            if result['top8_fourier_log_permuted'] is not None:
                print(f"  => Fourier (log-permuted): top8={result['top8_fourier_log_permuted']:.4f}  "
                      f"H_norm={result['fourier_log_entropy_norm']:.4f}  "
                      f"PR={result['fourier_log_participation_ratio']:.2f}")

        all_results[task] = task_results

    # ═══ Zusammenfassung ═══
    print("\n\n" + "=" * 70)
    print("  ERGEBNIS-ZUSAMMENFASSUNG: Addition vs. Multiplikation")
    print("=" * 70)

    for task in tasks:
        results = all_results[task]
        gen_steps = [r["test_generalized_step"] for r in results if r["test_generalized_step"] is not None]
        gaps = [r["grok_gap"] for r in results if r["grok_gap"] is not None]
        accs = [r["final_test_acc"] for r in results]
        f_nat = [r["top8_fourier_natural"] for r in results]
        f_nat_h = [r["fourier_natural_entropy_norm"] for r in results]
        f_nat_pr = [r["fourier_natural_participation_ratio"] for r in results]
        f_log = [r["top8_fourier_log_permuted"] for r in results if r["top8_fourier_log_permuted"] is not None]
        f_log_h = [r["fourier_log_entropy_norm"] for r in results if r["fourier_log_entropy_norm"] is not None]
        f_log_pr = [r["fourier_log_participation_ratio"] for r in results if r["fourier_log_participation_ratio"] is not None]

        grokked = len(gen_steps)
        print(f"\n  Task: {task.upper()} — {grokked}/{len(results)} Runs grokked")
        if gen_steps:
            print(f"    Generalization step: {np.mean(gen_steps):.0f} +/- {np.std(gen_steps):.0f}  "
                  f"(quantisiert auf Log-Gitter, siehe *_step_interval je Run)")
            print(f"    Grok gap:            {np.mean(gaps):.0f} +/- {np.std(gaps):.0f}")
        print(f"    Final test acc:      {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"    Fourier (nat)  top8={np.mean(f_nat):.4f}+/-{np.std(f_nat):.4f}  "
              f"H_norm={np.mean(f_nat_h):.4f}+/-{np.std(f_nat_h):.4f}  "
              f"PR={np.mean(f_nat_pr):.2f}+/-{np.std(f_nat_pr):.2f}")
        if f_log:
            print(f"    Fourier (log)  top8={np.mean(f_log):.4f}+/-{np.std(f_log):.4f}  "
                  f"H_norm={np.mean(f_log_h):.4f}+/-{np.std(f_log_h):.4f}  "
                  f"PR={np.mean(f_log_pr):.2f}+/-{np.std(f_log_pr):.2f}")

    # Ergebnisse als JSON speichern
    out_path = Path("add_vs_mul_results.json")
    save_data = {}
    for task in tasks:
        save_data[task] = [{k: v for k, v in r.items() if k != "curves"}
                           for r in all_results[task]]
    out_path.write_text(json.dumps(save_data, indent=2))
    print(f"\n  [Gespeichert] {out_path}")


if __name__ == "__main__":
    main()
