"""One-off edit script (pass 2026-09-03) for docs/sources/khanh2026_at_grok_not_converged.md.

Applies four additions to the existing verified note; every anchor must match exactly once.
Kept next to the raw source text for provenance; not part of the analysis code.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # .../Claude/grokverse
NOTE = ROOT / "docs" / "sources" / "khanh2026_at_grok_not_converged.md"
RAW_TXT = ROOT / "docs" / "sources" / "_raw" / "khanh_2607.06639.txt"

s = NOTE.read_text(encoding="utf-8")
raw = RAW_TXT.read_text(encoding="utf-8")
print("raw 'top-' occurrences:", len(re.findall(r"top-", raw)),
      "| '1/\\sqrt{d}':", raw.count(r"1/\sqrt{d}"),
      "| 'median $3.3':", raw.count("median $3.3"))

# 1. header row -------------------------------------------------------------
lines = s.split("\n")
idx = [i for i, l in enumerate(lines) if l.startswith("| verified |")]
assert len(idx) == 1, idx
hdr = (
    "| updated | [pass 2026-09-03] third independent reading for the run-format v2 / 25k-budget decision: "
    "`abs` and `html` re-fetched with ten section-specific WebFetch prompts; raw HTML re-downloaded with curl "
    "(324,240 bytes, same size as the verifier's copy) to `docs/sources/_raw/khanh_2607.06639.html` and converted "
    "to text with math `alttext` kept (`docs/sources/_raw/khanh_2607.06639.txt`). Every number and quote below was "
    "re-checked against that text; no correction to the 2026-09-02 content was needed. Two errors appeared only in "
    "the WebFetch summaries, not in this note — \"1/d initialization\" (raw HTML: `$1/\\sqrt{d}$`, Sec. 6) and "
    "Table 3 ε=0.05 \"1.46 / 1.42\" (raw HTML: add 1.36, mult 1.46) — so summaries must not be quoted without the "
    "raw text. Additions are marked \"[pass 2026-09-03]\": §6.2b (budget check against the *measured* `arch25k` "
    "crossings), §6.3 item 2 (the PREREG_BRIEF convergence rule vs. the paper's criterion), §7 ledger. |"
)
lines.insert(idx[0] + 1, hdr)
s = "\n".join(lines)

# 2. §6.2b before §6.3 -------------------------------------------------------
anchor = "\n### 6.3 What a GROKVERSE study must report"
assert s.count(anchor) == 1
sec62b = r"""
### 6.2b The same check against the measured `arch25k` crossings `[OWN COMPUTATION, pass 2026-09-03]`

Input: `training/runs/{mlp,txf}_add_p113_wd1.0_frac0.3_seed{0..9}_arch25k/run.json` →
`transitions.primary.generalization.first_crossing_step` (test acc ≥ 0.95, dense evaluation every 25 steps, so each
crossing is an interval `(step−25, step]`). Run metadata only — used here for budget adequacy, not as a result; the
paired timing comparison belongs to the statistics plan and is deliberately not made here. At the time of this
computation all 10 MLP runs and transformer seeds 0–7 had `steps_completed = 25000`; transformer seeds 8–9 were at
step 14,515 and still running (their crossings, at 8,125 and 5,825, are already fixed). Coefficients are the
paper's lag/$T_{\mathrm{grok}}$ values (0.63 un-normalized transformer, Sec. 7.2; 1.00 MLP default ε=0.10, Table 2;
1.46 the largest value in Table 3, ε=0.05 on multiplication), applied as $T_{\mathrm{compress}} = (1+c)\,T_{\mathrm{gen}}$.

| | MLP, seeds 0–9 | transformer (no LayerNorm), seeds 0–9 |
|---|---|---|
| $T_{\mathrm{gen}}$ (0.95), per seed | 9125, 10075, 9175, 8150, 8650, 9375, 9900, 9200, 9800, 9300 | 7975, 6250, 8125, 7425, 10275, 5625, 6325, 7750, 8125, 5825 |
| median / min / max | 9250 / 8150 / 10075 | 7587.5 / 5625 / 10275 |
| 25000 / $T_{\mathrm{gen}}$, median (min) | 2.70 (2.48) | 3.30 (2.43) |
| post-generalization window 25000 − $T_{\mathrm{gen}}$, min / median / max | 14925 / 15750 / 16850 | 14725 / 17412 / 19375 |
| predicted $T_{\mathrm{compress}}$ at c = 0.63, median / max | 15077 / 16422 | 12367 / 16748 |
| predicted $T_{\mathrm{compress}}$ at c = 1.00, median / max (seeds > 20000) | 18500 / 20150 (1: seed 1) | 15175 / 20550 (1: seed 4) |
| predicted $T_{\mathrm{compress}}$ at c = 1.46, median / max (seeds > 25000) | 22755 / 24784 (0; seed 1 has 216 steps of margin) | 18665 / 25276 (1: seed 4, censored) |
| windows shorter than the paper's absolute MLP lag (17k / 18k, Table 2) | 10 of 10 / 10 of 10 | 3 of 10 / 6 of 10 |
| windows shorter than the tolerance-free bound $10^{4}$ | 0 of 10 | 0 of 10 |

Reading, restricted to what the paper licenses (ratios, direction, and the $\geq 10^{4}$ bound; Sec. 5, Sec. 6.3):

1. The tolerance-free claim is cleared by every seed of both architectures: no post-generalization window is
   shorter than $10^{4}$ steps (minimum 14,725).
2. At the paper's default tolerance (c = 1.00, the MLP coefficient) all 20 seeds are predicted to settle inside
   the budget, but the slowest seed of each architecture lands between 20k and 25k — i.e. inside the last fifth of
   training that the PREREG_BRIEF convergence rule (§6.3 item 2) uses as its "already flat" window. For those seeds
   the step-20000 checkpoint would be read while the metric is, by this prediction, still moving.
3. At the strict tolerance (c = 1.46) one transformer seed (seed 4, $T_{\mathrm{gen}}$ = 10,275, the latest crossing
   in the matrix) is predicted to settle 276 steps *after* the budget and would be censored; the slowest MLP seed
   clears it by 216 steps. The strict tolerance is therefore the one at which the censoring flag (§6.3 item 3)
   must be expected to fire, and the sensitivity pair ε = 0.05 / 0.20 is not optional.
4. The paper's absolute MLP lags (17,000–18,000 steps at $p=59$, norm clamp, $d=128$, $H=256$, GELU) exceed every MLP
   post-generalization window in the matrix (max 16,850). Absolute transfer is not expected (different $p$, width,
   activation and protocol), but this is exactly why the study cannot borrow the paper's numbers and must date
   $T_{\mathrm{settle}}$ on its own trajectories, per seed and per metric.
5. The 0.63 coefficient — the only transformer number — leaves ≥ 8,250 steps of margin for every transformer seed.
   If the paper's ordering transfers (un-normalized transformer settles earlier than the MLP relative to its own
   $T_{\mathrm{gen}}$), the event-matched comparison at the crossing is the one most exposed to the timing
   confound (§6.1 item 2), and the budget-matched comparison at 25k is the safer of the two declared points for
   the transformer and the more marginal of the two for the MLP.

Conclusion unchanged from §6.2, now with the real crossings: 25k is adequate for the primary block at the default
tolerance provided the per-seed settling test is run and reported; it is marginal at the strict tolerance for the
slowest seed of each architecture, and the "converged at budget" label must come from the test, never from the
budget.
"""
s = s.replace(anchor, sec62b + anchor)

# 3. §6.3 item 2 update ------------------------------------------------------
anchor2 = 'otherwise the run is "floor not plateaued". (new)'
assert s.count(anchor2) == 1
add2 = anchor2 + (
    "\n   [pass 2026-09-03] `docs/dev/PREREG_BRIEF.md` (\"Setting\", row \"measurement points\") has since fixed a rule: "
    "\"A metric is called \\\"converged at budget\\\" only if its change between the step-20000 and step-25000 checkpoints "
    "is below 5 % of its value (Khanh 2607.06639); otherwise \\\"final (budget), not converged\\\"\". Two things the "
    "citation should not be read to imply `[OWN OBSERVATION]`: (a) the paper prints no plateau formula "
    "(`[NOT FOUND IN SOURCE]`, §3.4 above), so the 20k→25k / 5 % rule is GROKVERSE's own and should be cited as "
    "*motivated by* the paper, not *taken from* it; (b) the paper's tolerance is normalised by the at-grok-to-floor "
    "**drop** (\"ε=0.1 of the at-grok-to-floor drop\", Sec. 3.4) and its floor is a window median (\"computed over the "
    "final tenth of training\", Sec. 6.4; floor frac 0.05 / 0.20 in Table 3), whereas the PREREG rule is normalised by "
    "the metric's **value** and compares two single checkpoints one fifth of the budget apart. The two criteria "
    "diverge in opposite directions depending on the metric: for a concentration that moves, say, 0.70→0.76 between "
    "the crossing and the floor, 5 % of value (≈0.038) is ≈60 % of the drop — six times looser than the paper's ε; "
    "for an effective rank falling 50→7, 5 % of the floor value (≈0.35) is <1 % of the drop — ten times stricter. "
    "Recommendation: keep the PREREG rule as the pre-registered *label* rule (it is fixed and must not be changed post "
    "hoc), but report the paper-style quantities next to it for every metric — $T_{\\mathrm{settle}}(\\varepsilon)$ "
    "with ε ∈ {0.05, 0.10, 0.20} of the drop, floor = median over the final tenth (sensitivity 0.05 / 0.20) — so that "
    "a reader can compare GROKVERSE to the paper on the paper's own definition. The 20k checkpoint is also inside the "
    "predicted settling region of the slowest seeds at the default tolerance (§6.2b item 2), so a \"converged at "
    "budget\" verdict for those seeds is the case where the two definitions are most likely to disagree; if they do, "
    "say so rather than pick one."
)
s = s.replace(anchor2, add2)

# 4. ledger -------------------------------------------------------------------
s = s.rstrip("\n") + r"""

[pass 2026-09-03] Third reading, independent of the two above: `abs` and `html` re-fetched (ten section-specific
prompts), raw HTML re-downloaded with curl (324,240 bytes) and converted with `alttext` kept
(`docs/sources/_raw/khanh_2607.06639.{html,txt}`). Re-confirmed in the raw text: every number in §0–§5, the full
Table 3 (ε rows: add 24750/1.36, 18000/1.03, 11500/0.65; mult 25500/1.46, 17000/1.03, 10500/0.67), the arm-B
initialization `$1/\sqrt{d}$` (Sec. 6), "final tenth of training" (Sec. 1.2 and Sec. 6.4), "dense post-grok logging"
without a cadence (Sec. 7), "few seeds" without a count (Sec. 6.3, Sec. 10.3), the absence of the string "top-"
anywhere in the paper. Two errors were produced by the WebFetch summariser and rejected against the raw text
("1/d initialization"; Table 3 ε=0.05 "1.46 / 1.42"). No correction to the 2026-09-02 text. Additions: header row,
§6.2b (measured `arch25k` crossings vs. the paper's lag coefficients, computed with the project Python from
`run.json`), §6.3 item 2 (PREREG_BRIEF convergence rule vs. the paper's criterion, flagged as `[OWN OBSERVATION]`).
Nothing from memory.
"""
NOTE.write_text(s, encoding="utf-8")
print("written; lines:", s.count("\n") + 1, "| '2026-09-03' occurrences:", s.count("2026-09-03"))
