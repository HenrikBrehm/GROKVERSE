"""Named restricted/excluded protocols for the Nanda 2023 progress measures.

WHY THIS MODULE EXISTS — docs/RESEARCH_SPEC.md §3.1 [AUDIT]
-----------------------------------------------------------
``progress_measures.py`` originally built its 2D masks as the outer product of a
1D "keep" mask::

    mask_restr = keep_restr[:, None] & keep_restr[None, :]

That keeps **every pair** of key-frequency rows, including *cross-frequency*
blocks such as ``cos(w_18 a) * cos(w_15 b)``. The trig-identity circuit only ever
uses the **same** frequency on both axes::

    cos(w_k(a+b)) = cos(w_k a)cos(w_k b) - sin(w_k a)sin(w_k b)
    sin(w_k(a+b)) = sin(w_k a)cos(w_k b) + cos(w_k a)sin(w_k b)

so the outer product gives the "circuit" far more freedom than the hypothesis
allows, and the matching excluded mask deletes far more of the logit tensor than
the hypothesis targets. Both biases push toward the reported conclusion. This
module implements the variants **side by side, named**, so the width of the mask
is a stated choice instead of an accident.

The 2D coefficient plane
------------------------
``fourier.fourier_basis`` returns rows: index 0 = constant, then for k = 1.. the
pair ``(cos_k, sin_k)`` at rows ``1+2(k-1)``, ``2+2(k-1)``. A logit tensor
``L[a, b, c]`` is transformed over the two *input* axes into ``Lhat[i, j, c]``
where ``i`` indexes the a-axis mode and ``j`` the b-axis mode.

For one key frequency k the four "same-frequency" entries are

    A = Lhat[cos_k, cos_k]   B = Lhat[sin_k, sin_k]
    C = Lhat[sin_k, cos_k]   D = Lhat[cos_k, sin_k]

and since the cos_k and sin_k basis rows carry the same normalization, the four
functions of (a, b) that span this block decompose into two orthogonal pairs::

    cos(w_k(a+b)) ~ (A, B, C, D) = (1, -1, 0, 0)      "sum" directions
    sin(w_k(a+b)) ~ (0,  0, 1, 1)
    cos(w_k(a-b)) ~ (1,  1, 0, 0)                     "difference" directions
    sin(w_k(a-b)) ~ (0,  0, 1, -1)

**The sum directions are a 2D subspace inside the 4D block, not two coordinates
of it.** ``sum_directions_only`` is therefore a genuine projection, not a
boolean mask — a mask cannot express it. That is why every protocol here is a
linear *operator* on the coefficient plane rather than an array of booleans.

Variants (component counts for p=113, 8 key frequencies — asserted in tests)
---------------------------------------------------------------------------
===========================  =========  ===========  =================================
name (report name)           kept       removed      what it assumes
===========================  =========  ===========  =================================
``legacy_broad_mask``        289        3360 (26.3%) every pair of key rows, cross-freq
``same_frequency_block``     33         32           same frequency on both axes
  (reported as
  ``paper_literal_2x2_block``)
``sum_directions_only``      17         16           only cos/sin(w_k(a+b))
``diff_directions_only``     17         16           only cos/sin(w_k(a-b)) — CONTROL
``nanda_exact``              17         16           the published operator (audit §4)
===========================  =========  ===========  =================================

``nanda_exact`` — what is reconstructed and what is not
-------------------------------------------------------
``docs/MASK_PROTOCOL_AUDIT.md`` §4 resolves the *operator*: restricted keeps the
constant plus, per key ``k``, the projection onto
``span{cos(w_k(a+b)), sin(w_k(a+b))}``; excluded is ``L - P_S(L)`` with everything
else, constant included, untouched. That is bit-for-bit the same operator as
``SumDirectionsOnly`` — the Colab's ``get_component_cos_xpy`` /
``get_component_sin_xpy`` projections, quoted in
``docs/sources/nanda2023_progress_measures.md`` §5.

The one item that stays **[NOT FOUND IN SOURCE]** is the DATA SPLIT of the
paper's own restricted-loss figure (audit §1 row 8, §7 item 1). The excluded
split *is* stated ("We measure this on the training data", paper §5.1). So
``NandaExact`` implements the operator but **refuses to be constructed without a
declared split**: there is no default, because a default would silently pick the
convention the paper never states. ``nanda_exact_restricted_loss`` therefore
builds it once per split and returns all three, labelling ``test`` as the quoted
convention (TransformerLens demo) and recording the paper's split as
``[NOT FOUND IN SOURCE]``.

Consequence, pinned by ``test_core.py``: ``build_protocol("nanda_exact", p, keys)``
raises — the split argument is missing. ``build_protocol("nanda_exact", p, keys,
split="all")`` works.

``diff_directions_only`` is the **mirror control**, not a hypothesis: no source
ever keeps or deletes the ``(a-b)`` directions. It exists so that "the sum
directions carry the circuit" is a measurement against a same-sized alternative
rather than an assertion (INTERFACES §5, §10).

Convention shared by all variants
---------------------------------
* ``restrict`` keeps the constant term **plus** the circuit subspace S.
* ``exclude`` removes S and keeps everything else, constant included.

so ``n_kept == n_removed + 1`` for the exact variants. ``legacy_broad_mask``
breaks that relation (289 vs 3360) — its restricted and excluded masks are not
complements of one another. That inconsistency is a property of the legacy
implementation, reproduced here on purpose so it can be compared, not hidden.
"""
from __future__ import annotations

import numpy as np

from .fourier import fourier_basis

#: The three data splits any restricted/excluded number can be quoted on.
#: ``all`` is the full (a, b) grid — GROKVERSE's legacy convention, and the
#: Colab's ``mode='all'``; ``train`` is the paper's excluded-loss split;
#: ``test`` is the TransformerLens demo's restricted-loss split.
SPLITS: tuple[str, ...] = ("test", "train", "all")


def _rows(k: int) -> tuple[int, int]:
    """(cos_k row, sin_k row) in the fourier_basis layout."""
    return 1 + 2 * (k - 1), 2 + 2 * (k - 1)


def _validate(p: int, key_freqs) -> tuple[int, ...]:
    if p % 2 == 0:
        raise ValueError(f"protocols require odd p (got {p})")
    half = (p - 1) // 2
    keys = tuple(int(k) for k in key_freqs)
    if not keys:
        raise ValueError("key_freqs is empty — no circuit to restrict to")
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate key frequencies: {keys}")
    bad = [k for k in keys if not 1 <= k <= half]
    if bad:
        raise ValueError(f"key frequencies {bad} outside 1..{half} for p={p}")
    return keys


class Protocol:
    """A named restricted/excluded pair of linear operators on ``Lhat``.

    Subclasses implement ``restrict``/``exclude``; both take and return an array
    of shape ``[p, p, n_classes]`` and must not mutate their input.
    """

    name: str = "abstract"

    #: Name used in reports and JSON output. Identical to ``name`` except where
    #: the audit renames a variant (``same_frequency_block`` is reported as
    #: ``paper_literal_2x2_block``, audit §3). Both names reach the same class
    #: through ``build_protocol``; the old one is never broken.
    report_name: str = "abstract"

    def __init__(self, p: int, key_freqs) -> None:
        self.p = p
        self.key_freqs = _validate(p, key_freqs)

    def restrict(self, Lhat: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def exclude(self, Lhat: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @property
    def n_kept(self) -> int:  # pragma: no cover
        raise NotImplementedError

    @property
    def n_removed(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "protocol": self.name,
            "report_name": self.report_name,
            "p": self.p,
            "key_frequencies": list(self.key_freqs),
            "n_components_kept_by_restricted": self.n_kept,
            "n_components_removed_by_excluded": self.n_removed,
            "n_components_total": self.p ** 2,
            "keeps_cross_frequency_blocks": self.keeps_cross_frequency(),
        }

    def keeps_cross_frequency(self) -> bool:
        """Does ``restrict`` pass any component with a DIFFERENT frequency on
        the two axes? True only for the legacy mask. Probed by construction:
        feed in a single cross-frequency unit component and see if it survives.
        """
        if len(self.key_freqs) < 2:
            return False
        k1, k2 = self.key_freqs[0], self.key_freqs[1]
        probe = np.zeros((self.p, self.p, 1))
        probe[_rows(k1)[0], _rows(k2)[0], 0] = 1.0
        return bool(np.abs(self.restrict(probe)).max() > 1e-12)


class _MaskProtocol(Protocol):
    """Protocol expressible as elementwise multiplication by a boolean mask."""

    def __init__(self, p: int, key_freqs, keep: np.ndarray, remove: np.ndarray) -> None:
        super().__init__(p, key_freqs)
        self._keep = keep
        self._remove = remove

    def restrict(self, Lhat: np.ndarray) -> np.ndarray:
        return Lhat * self._keep[:, :, None]

    def exclude(self, Lhat: np.ndarray) -> np.ndarray:
        return Lhat * (~self._remove)[:, :, None]

    @property
    def n_kept(self) -> int:
        return int(self._keep.sum())

    @property
    def n_removed(self) -> int:
        return int(self._remove.sum())


class LegacyBroadMask(_MaskProtocol):
    """The original implementation: outer product of a 1D key-row mask.

    Kept runnable so the old numbers can be reproduced and compared — but it is
    NOT a reproduction of Nanda et al. and must never be labelled as one.
    """

    name = "legacy_broad_mask"
    report_name = "legacy_broad_mask"

    def __init__(self, p: int, key_freqs) -> None:
        keys = _validate(p, key_freqs)
        keep_1d = np.zeros(p, dtype=bool)
        is_key = np.zeros(p, dtype=bool)
        keep_1d[0] = True                      # constant always kept
        for k in keys:
            for r in _rows(k):
                keep_1d[r] = True
                is_key[r] = True
        keep = keep_1d[:, None] & keep_1d[None, :]
        remove = is_key[:, None] | is_key[None, :]
        super().__init__(p, keys, keep, remove)


class SameFrequencyBlock(_MaskProtocol):
    """Constant + the full 2x2 (cos,sin) x (cos,sin) block of each key frequency.

    Drops cross-frequency blocks but keeps both the sum AND difference
    directions of each block — i.e. it also permits ``cos(w_k(a-b))``, which the
    addition circuit has no use for. The intermediate variant between the legacy
    mask and ``sum_directions_only``.
    """

    name = "same_frequency_block"
    #: audit §3: reported as the paper-literal reading; the class name is kept.
    report_name = "paper_literal_2x2_block"

    def __init__(self, p: int, key_freqs) -> None:
        keys = _validate(p, key_freqs)
        keep = np.zeros((p, p), dtype=bool)
        remove = np.zeros((p, p), dtype=bool)
        keep[0, 0] = True                      # constant
        for k in keys:
            c, s = _rows(k)
            for i in (c, s):
                for j in (c, s):
                    keep[i, j] = True
                    remove[i, j] = True        # excluded keeps the constant
        super().__init__(p, keys, keep, remove)


class SumDirectionsOnly(Protocol):
    """Constant + only the ``cos/sin(w_k(a+b))`` directions of each key block.

    The tightest variant: exactly the components the angle-addition identity
    produces. Implemented as an orthogonal projection inside each 4D block
    (see the module docstring) — the sum directions are a 2D *subspace*, not two
    coordinates, so no boolean mask can express this.
    """

    name = "sum_directions_only"
    report_name = "sum_directions_only"

    def _project(self, Lhat: np.ndarray) -> np.ndarray:
        """Orthogonal projection onto span{cos(w_k(a+b)), sin(w_k(a+b))}_k.

        Excludes the constant term; ``restrict`` adds it back.
        """
        out = np.zeros_like(Lhat)
        for k in self.key_freqs:
            c, s = _rows(k)
            # (A, B) onto (1, -1)/sqrt2  ->  ((A-B)/2, -(A-B)/2)
            u = (Lhat[c, c] - Lhat[s, s]) / 2.0
            out[c, c], out[s, s] = u, -u
            # (C, D) onto (1, 1)/sqrt2   ->  ((C+D)/2, (C+D)/2)
            v = (Lhat[s, c] + Lhat[c, s]) / 2.0
            out[s, c], out[c, s] = v, v
        return out

    def restrict(self, Lhat: np.ndarray) -> np.ndarray:
        out = self._project(Lhat)
        out[0, 0] = Lhat[0, 0]
        return out

    def exclude(self, Lhat: np.ndarray) -> np.ndarray:
        return Lhat - self._project(Lhat)

    @property
    def n_kept(self) -> int:
        return 1 + 2 * len(self.key_freqs)     # constant + 2 directions per freq

    @property
    def n_removed(self) -> int:
        return 2 * len(self.key_freqs)


class DiffDirectionsOnly(Protocol):
    """Constant + only the ``cos/sin(w_k(a-b))`` directions of each key block.

    The exact **mirror** of ``SumDirectionsOnly``, and a CONTROL rather than a
    hypothesis: no source ever keeps or deletes the difference directions, and
    the angle-addition identity has no use for them. It exists so that
    "the ``(a+b)`` directions carry the circuit" is measured against a
    same-sized alternative subspace (17 kept / 16 removed, identical to
    ``sum_directions_only``) instead of asserted. Used by
    ``mlp_mechanism.activation_analysis`` for ``diff_direction_share``
    (INTERFACES §5) and by ``per_frequency_shares`` below.

    Projection inside each 4D block (module docstring):
    ``cos(w_k(a-b)) ~ (A, B, C, D) = (1, 1, 0, 0)`` and
    ``sin(w_k(a-b)) ~ (0, 0, 1, -1)``.
    """

    name = "diff_directions_only"
    report_name = "diff_directions_only"

    def _project(self, Lhat: np.ndarray) -> np.ndarray:
        """Orthogonal projection onto span{cos(w_k(a-b)), sin(w_k(a-b))}_k.

        Excludes the constant term; ``restrict`` adds it back.
        """
        out = np.zeros_like(Lhat)
        for k in self.key_freqs:
            c, s = _rows(k)
            # (A, B) onto (1, 1)/sqrt2   ->  ((A+B)/2, (A+B)/2)
            u = (Lhat[c, c] + Lhat[s, s]) / 2.0
            out[c, c], out[s, s] = u, u
            # (C, D) onto (1, -1)/sqrt2  ->  ((C-D)/2, -(C-D)/2)
            v = (Lhat[s, c] - Lhat[c, s]) / 2.0
            out[s, c], out[c, s] = v, -v
        return out

    def restrict(self, Lhat: np.ndarray) -> np.ndarray:
        out = self._project(Lhat)
        out[0, 0] = Lhat[0, 0]
        return out

    def exclude(self, Lhat: np.ndarray) -> np.ndarray:
        return Lhat - self._project(Lhat)

    @property
    def n_kept(self) -> int:
        return 1 + 2 * len(self.key_freqs)     # constant + 2 directions per freq

    @property
    def n_removed(self) -> int:
        return 2 * len(self.key_freqs)


class NandaExact(SumDirectionsOnly):
    """The published operator (Nanda et al. 2023 §5.1) with a DECLARED split.

    RECONSTRUCTED (docs/MASK_PROTOCOL_AUDIT.md §4, quoted in
    docs/sources/nanda2023_progress_measures.md §5–§6):

    * restricted = the constant component **plus**, per key ``k``, the orthogonal
      projection onto ``span{cos(w_k(a+b)), sin(w_k(a+b))}``. Re-inserting
      ``Lhat[0, 0]`` is exactly the Colab's ``bias_correction=True`` (the
      projected logits have zero grid mean, so adding the per-class mean of the
      original logits *is* the constant component).
    * excluded = ``L - P_S(L)`` with ``S`` the same sum-direction subspace over
      all key frequencies at once; the constant and every other component are
      untouched, and there is no bias correction.
    * no cross-frequency products, no difference directions, no linear
      ``const x cos_k`` terms.

    This is the same operator as ``SumDirectionsOnly`` — which is why this class
    subclasses it rather than copying the projection. The audit's §3 verdict on
    ``sum_directions_only`` is "matches the operator the released code
    implements"; the separate name exists so a report can say *published
    protocol* where that is what is meant.

    **[NOT FOUND IN SOURCE]** — the DATA SPLIT of the paper's own restricted-loss
    curve (audit §1 row 8, §7 item 1). The paper states the split for excluded
    loss only ("We measure this on the training data", §5.1). The released Colab
    plots restricted on ``mode='all'`` *and* ``mode='train'``; the TransformerLens
    demo uses ``test``. Rather than guess, this class takes the split as a
    REQUIRED constructor argument and every consumer records it:
    ``nanda_exact_restricted_loss`` returns all three, labelled.

    Also not reproduced here, and deliberately: the paper's key-frequency count
    is *measured* per model (5 mainline, 3–4 other seeds) by a rule whose numeric
    threshold is ``[NOT FOUND IN SOURCE]``. This class takes whatever
    ``key_freqs`` it is given; choosing them is ``analysis.key_frequencies``
    (INTERFACES §4, primary rule ``nanda``), never a fixed top-8.
    """

    name = "nanda_exact"
    report_name = "nanda_exact"

    #: What the operator reproduces and what it cannot — echoed into every
    #: result dict so a number can never travel without its provenance.
    PROVENANCE: dict[str, str] = {
        "operator": "reproduced (audit §4; Colab get_component_cos_xpy/sin_xpy)",
        "constant_term": "kept by restricted (= bias_correction=True); untouched by excluded",
        "excluded_split": "train — stated in the paper (§5.1 'We measure this on the training data')",
        "restricted_split_in_paper": "[NOT FOUND IN SOURCE]",
        "restricted_split_quoted_here": "test, following the TransformerLens demo "
                                        "(Grokking_Demo.ipynb get_restricted_loss)",
        "key_frequency_threshold": "[NOT FOUND IN SOURCE] — the paper publishes no numeric "
                                   "threshold for 'nontrivial coefficients' (audit §7 item 2)",
    }

    def __init__(self, p: int, key_freqs, split: str) -> None:
        """``split`` is REQUIRED and has no default — see the class docstring.

        ``build_protocol("nanda_exact", p, keys)`` therefore raises ``TypeError``;
        ``build_protocol("nanda_exact", p, keys, split="all")`` works. The split
        does not change the operator (``restrict``/``exclude`` act on ``Lhat``
        alone); it declares which cells the resulting loss is read on, which is
        the one item the audit marks unresolved.
        """
        super().__init__(p, key_freqs)
        if split not in SPLITS:
            raise ValueError(
                f"nanda_exact needs an explicit split from {SPLITS}, got {split!r}. "
                "The paper's restricted-loss split is [NOT FOUND IN SOURCE] "
                "(MASK_PROTOCOL_AUDIT §4), so there is no default to fall back on.")
        self.split = split

    def describe(self) -> dict:
        return {**super().describe(), "split": self.split, "provenance": dict(self.PROVENANCE)}


_REGISTRY: dict[str, type[Protocol]] = {
    LegacyBroadMask.name: LegacyBroadMask,
    SameFrequencyBlock.name: SameFrequencyBlock,
    SumDirectionsOnly.name: SumDirectionsOnly,
    DiffDirectionsOnly.name: DiffDirectionsOnly,
    NandaExact.name: NandaExact,
    # audit §3 rename, reports only — the class keeps its own name.
    SameFrequencyBlock.report_name: SameFrequencyBlock,
}

#: Report name -> class name, for callers that need to resolve either spelling.
ALIASES: dict[str, str] = {SameFrequencyBlock.report_name: SameFrequencyBlock.name}

#: The three variants the legacy re-train path (``progress_measures.compute``)
#: and ``test_core.py`` iterate. FROZEN: test_core pins this tuple's membership
#: and the ``--protocols`` CLI choices come from it. New variants go into
#: ``ALL_IMPLEMENTED`` / ``SPLIT_AGNOSTIC`` below, not here.
IMPLEMENTED: tuple[str, ...] = (
    LegacyBroadMask.name, SameFrequencyBlock.name, SumDirectionsOnly.name,
)

#: Every implemented variant under the name reports use, tightest last.
#: ``nanda_exact`` is here but NOT in ``SPLIT_AGNOSTIC``: it cannot be built
#: without a declared split (see ``NandaExact``).
ALL_IMPLEMENTED: tuple[str, ...] = (
    LegacyBroadMask.report_name, SameFrequencyBlock.report_name,
    DiffDirectionsOnly.report_name, SumDirectionsOnly.report_name,
    NandaExact.report_name,
)

#: Those of ``ALL_IMPLEMENTED`` that ``build_protocol(name, p, keys)`` builds
#: with no further arguments.
SPLIT_AGNOSTIC: tuple[str, ...] = tuple(
    n for n in ALL_IMPLEMENTED if n != NandaExact.report_name)

ALL_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def build_protocol(name: str, p: int, key_freqs, **kwargs) -> Protocol:
    """Build a protocol by class name or report name.

    ``kwargs`` are forwarded to the constructor; only ``nanda_exact`` uses them
    (``split=``), and for it they are mandatory — a missing ``split`` raises
    ``TypeError`` by design (see ``NandaExact``).
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown protocol {name!r}; choices: {list(_REGISTRY)}")
    return _REGISTRY[name](p, key_freqs, **kwargs)


def operator_trace(op, p: int) -> float:
    """Trace of a linear operator on the [p, p] coefficient plane.

    The trace of a projection equals the dimension of its image, so this gives a
    numerical cross-check of the analytic ``n_kept``/``n_removed`` counts that
    works for mask and projection protocols alike. Feeds the operator one unit
    basis component per (i, j) — batched over j through the class axis — and
    sums the diagonal entries.
    """
    tr = 0.0
    for i in range(p):
        E = np.zeros((p, p, p))
        E[i, np.arange(p), np.arange(p)] = 1.0   # channel j holds component (i, j)
        out = op(E)
        tr += float(out[i, np.arange(p), np.arange(p)].sum())
    return tr


# --------------------------------------------------------------------------- #
# Per-frequency power bookkeeping (INTERFACES §10 "per_frequency option")      #
# --------------------------------------------------------------------------- #
def per_frequency_shares(Lhat: np.ndarray, key_freqs) -> dict:
    """Share of the total NON-CONSTANT power sitting in each key k's directions.

    ``Lhat`` is the 2D-transformed tensor ``[p, p, n_classes]``
    (``progress_measures.fwd2d``). Power is summed over the class axis. For each
    key ``k`` the 4D same-frequency block ``{A, B, C, D}`` splits orthogonally
    into the two sum directions ``cos/sin(w_k(a+b))`` and the two difference
    directions ``cos/sin(w_k(a-b))``, so ``sum_power + diff_power ==
    block_power`` exactly (the identity ``A^2 + B^2 = 2u^2 + 2u'^2`` with
    ``u = (A-B)/2``, ``u' = (A+B)/2``).

    Blocks of distinct ``k`` are disjoint and none of them touches the constant,
    so the shares partition the non-constant power::

        sum over k of (sum_share + diff_share)  +  other_share  ==  1

    ``other_share`` is everything the key blocks do not reach: non-key
    frequencies, cross-frequency products, and the ``const x cos_k`` linear
    terms. It is reported, not silently dropped — that residual is the whole
    difference between the exact protocols and ``legacy_broad_mask``.

    This is a DESCRIPTIVE decomposition. A large ``sum_share`` is evidence about
    where the logit power sits, never by itself a claim that the model computes
    the trig-identity circuit (INTERFACES §0: no interpretation in code).
    """
    Lhat = np.asarray(Lhat, dtype=np.float64)
    if Lhat.ndim != 3 or Lhat.shape[0] != Lhat.shape[1]:
        raise ValueError(f"Lhat must be [p, p, n_classes], got shape {Lhat.shape}")
    p = int(Lhat.shape[0])
    keys = _validate(p, key_freqs)

    total = float((Lhat ** 2).sum())
    const = float((Lhat[0, 0] ** 2).sum())
    nonconst = total - const
    if not np.isfinite(nonconst) or nonconst <= 0.0:
        raise ValueError(
            "Lhat carries no non-constant power — per-frequency shares are undefined "
            f"(total={total!r}, constant={const!r})")

    per: dict[str, dict] = {}
    sum_total = diff_total = 0.0
    for k in keys:
        c, s = _rows(k)
        A, B = Lhat[c, c], Lhat[s, s]
        C, D = Lhat[s, c], Lhat[c, s]
        u, v = (A - B) / 2.0, (C + D) / 2.0          # sum directions
        up, vp = (A + B) / 2.0, (C - D) / 2.0        # difference directions
        s_pow = float(2.0 * ((u ** 2).sum() + (v ** 2).sum()))
        d_pow = float(2.0 * ((up ** 2).sum() + (vp ** 2).sum()))
        block = float((A ** 2).sum() + (B ** 2).sum() + (C ** 2).sum() + (D ** 2).sum())
        per[str(int(k))] = {
            "frequency": int(k),
            "sum_power": s_pow, "diff_power": d_pow, "block_power": block,
            "sum_share": s_pow / nonconst, "diff_share": d_pow / nonconst,
            "block_share": block / nonconst,
            "sum_over_block": (s_pow / block) if block > 0 else None,
        }
        sum_total += s_pow
        diff_total += d_pow

    other = nonconst - sum_total - diff_total
    return {
        "p": p,
        "key_frequencies": [int(k) for k in keys],
        "total_power": total,
        "constant_power": const,
        "total_nonconstant_power": nonconst,
        "per_frequency": per,
        "sum_power_total": sum_total,
        "diff_power_total": diff_total,
        "other_power": other,
        "sum_share_total": sum_total / nonconst,
        "diff_share_total": diff_total / nonconst,
        "block_share_total": (sum_total + diff_total) / nonconst,
        "other_share": other / nonconst,
        # must be 1 to numerical precision — asserted in tests/test_mask_protocols.py
        "shares_sum": (sum_total + diff_total + other) / nonconst,
    }


# --------------------------------------------------------------------------- #
# Module-level restricted / excluded losses with an explicit, labelled split   #
# (master prompt §6.3, INTERFACES §10, MASK_PROTOCOL_AUDIT §4)                 #
# --------------------------------------------------------------------------- #
#: Split of the paper's own restricted-loss figure. Unresolved on purpose.
PAPER_RESTRICTED_SPLIT = "[NOT FOUND IN SOURCE]"
#: The split a single quoted "Nanda-style" restricted number follows here.
QUOTED_RESTRICTED_SPLIT = "test"
QUOTED_RESTRICTED_SPLIT_SOURCE = (
    "TransformerLens demos/Grokking_Demo.ipynb, get_restricted_loss (cell 131): "
    "loss_fn(restricted_logits[test_indices], test_labels). The paper itself states "
    "no split for restricted loss; the released Colab plots mode='all' and "
    "mode='train' and reports them as identical.")
#: The split the paper does state for excluded loss.
EXCLUDED_SPLIT = "train"
EXCLUDED_SPLIT_SOURCE = (
    "paper §5.1 'We measure this on the training data'; blog 'Then measure the train "
    "loss'; Colab excl_loss mode='train'; TransformerLens get_excluded_loss train_indices.")

_AUDIT_REF = "docs/MASK_PROTOCOL_AUDIT.md §4"
_SOURCE_REF = "docs/sources/nanda2023_progress_measures.md §5-§6"


def _transforms():
    """``(fwd2d, inv2d)`` from ``progress_measures``, imported lazily.

    ``progress_measures`` imports this module, so a module-level import would be
    circular. INTERFACES §0 pins those two functions as *the* 2D transform over
    ``(a, b)``, so they are not re-implemented here.
    """
    from .progress_measures import fwd2d, inv2d
    return fwd2d, inv2d


def _check_grid(logits: np.ndarray, cfg) -> np.ndarray:
    L = np.asarray(logits, dtype=np.float64)
    p = int(cfg.p)
    if L.ndim != 3 or L.shape[0] != p or L.shape[1] != p:
        raise ValueError(
            f"logits must be the full grid [p, p, n_classes] with p={p}, got {L.shape}")
    if getattr(cfg, "task", "add") != "add":
        raise ValueError(
            f"task {cfg.task!r}: the split evaluation labels cells (a+b) mod p "
            "(common.masked_ce_and_acc), which is defined for modular addition only")
    return L


def _all_split_losses(L: np.ndarray, cfg) -> dict:
    """CE and accuracy of a rebuilt logit grid on ``test``, ``train`` and ``all``."""
    from .common import masked_ce_and_acc, split_masks   # lazy: torch-dependent
    p = int(cfg.p)
    train_mask, test_mask = split_masks(cfg)
    masks = {"test": test_mask, "train": train_mask,
             "all": np.ones((p, p), dtype=bool)}
    out: dict[str, dict] = {}
    for name, m in masks.items():
        ce, acc = masked_ce_and_acc(L, m, p)
        out[name] = {"loss": ce, "accuracy": acc, "n_cells": int(m.sum())}
    return out


def _apply(logits: np.ndarray, protocol: Protocol, which: str, cfg) -> np.ndarray:
    """Rebuild the logit grid after ``restrict`` or ``exclude``."""
    if which not in ("restricted", "excluded"):
        raise ValueError(f"which must be 'restricted' or 'excluded', got {which!r}")
    L = _check_grid(logits, cfg)
    fwd2d, inv2d = _transforms()
    Fb, _ = fourier_basis(int(cfg.p))
    Lhat = fwd2d(L, Fb)
    op = protocol.restrict if which == "restricted" else protocol.exclude
    return inv2d(op(Lhat), Fb)


def _record(protocol: Protocol, which: str, split: str, splits: dict,
            key_freqs, extra: dict) -> dict:
    """The shared return shape: ``{loss, accuracy, split, n_..., protocol, ...}``."""
    n = protocol.n_kept if which == "restricted" else protocol.n_removed
    return {
        "protocol": protocol.report_name,
        "which": which,
        "split": split,
        "loss": splits[split]["loss"],
        "accuracy": splits[split]["accuracy"],
        "n_components_kept_or_removed": int(n),
        "n_components_kept_by_restricted": int(protocol.n_kept),
        "n_components_removed_by_excluded": int(protocol.n_removed),
        "n_components_total": int(protocol.p ** 2),
        "n_cells_evaluated": splits[split]["n_cells"],
        "key_frequencies": [int(k) for k in key_freqs],
        "splits": splits,
        "keeps_cross_frequency_blocks": protocol.keeps_cross_frequency(),
        **extra,
    }


def nanda_exact_restricted_loss(logits: np.ndarray, key_freqs, cfg) -> dict:
    """Restricted loss under the published operator, reported on ALL THREE splits.

    Operator: constant + per key ``k`` the projection onto
    ``span{cos(w_k(a+b)), sin(w_k(a+b))}`` (``NandaExact``). The paper's own split
    is ``[NOT FOUND IN SOURCE]``, so the top-level ``loss``/``accuracy`` follow the
    **test** convention of the TransformerLens demo and say so; ``splits`` holds
    ``test``, ``train`` and ``all`` side by side. Nothing here picks the paper's
    split — the returned dict records that it is unstated.
    """
    pr = NandaExact(int(cfg.p), key_freqs, split=QUOTED_RESTRICTED_SPLIT)
    splits = _all_split_losses(_apply(logits, pr, "restricted", cfg), cfg)
    return _record(pr, "restricted", QUOTED_RESTRICTED_SPLIT, splits, pr.key_freqs, {
        "quoted_split": QUOTED_RESTRICTED_SPLIT,
        "quoted_split_source": QUOTED_RESTRICTED_SPLIT_SOURCE,
        "paper_split": PAPER_RESTRICTED_SPLIT,
        "paper_split_note": (
            "The paper states no split for restricted loss (audit §1 row 8). All three "
            "are reported; any single quoted number must name its split."),
        "is_our_variant": False,
        "provenance": dict(NandaExact.PROVENANCE),
        "audit_reference": _AUDIT_REF,
        "source_reference": _SOURCE_REF,
    })


def nanda_exact_excluded_loss(logits: np.ndarray, key_freqs, cfg,
                              per_frequency: bool = True) -> dict:
    """Excluded loss under the published operator, on the TRAINING pairs.

    ``L - P_S(L)`` with ``S`` the sum-direction subspace of **all** key
    frequencies at once (paper Fig. 7 / "full excluded loss"); the constant and
    every other component are untouched and there is no bias correction. The
    split is ``train`` and the paper does state it (§5.1).

    ``per_frequency`` additionally reports ``L - P_{S_k}(L)`` for each single
    ``k`` — the variant the released Colab (``excl_loss``, cell 216) actually
    plots and the paper shows in Fig. 15. The other two splits are reported
    alongside for comparison but the declared split stays ``train``.
    """
    pr = NandaExact(int(cfg.p), key_freqs, split=EXCLUDED_SPLIT)
    splits = _all_split_losses(_apply(logits, pr, "excluded", cfg), cfg)
    extra: dict = {
        "split_source": EXCLUDED_SPLIT_SOURCE,
        "is_our_variant": False,
        "bias_correction": False,
        "provenance": dict(NandaExact.PROVENANCE),
        "audit_reference": _AUDIT_REF,
        "source_reference": _SOURCE_REF,
        "note_projection_uses_all_data": (
            "The projection is fitted on the full grid before the train cells are "
            "selected (blog: 'technically also a function of the test data')."),
    }
    if per_frequency:
        single: dict[str, dict] = {}
        for k in pr.key_freqs:
            pk = NandaExact(int(cfg.p), [int(k)], split=EXCLUDED_SPLIT)
            sp = _all_split_losses(_apply(logits, pk, "excluded", cfg), cfg)
            single[str(int(k))] = {
                "frequency": int(k),
                "loss": sp[EXCLUDED_SPLIT]["loss"],
                "accuracy": sp[EXCLUDED_SPLIT]["accuracy"],
                "n_components_removed": int(pk.n_removed),
                "splits": sp,
            }
        extra["per_frequency"] = single
        extra["per_frequency_note"] = (
            "Colab excl_loss (cell 216) and paper Fig. 15 remove ONE key frequency at a "
            "time; the top-level number removes all of them at once (paper Fig. 7).")
    return _record(pr, "excluded", EXCLUDED_SPLIT, splits, pr.key_freqs, extra)


def full_grid_extension_restricted_loss(logits: np.ndarray, key_freqs, cfg) -> dict:
    """OUR variant: the exact component rule, read on the FULL (a, b) grid.

    Same operator as ``nanda_exact_restricted_loss``; only the split differs
    (``all`` instead of a held-out split). It exists so the new numbers can be
    compared against the legacy full-grid ones without confounding the mask
    change with the split change (audit §4). Never label it a reproduction.
    """
    pr = NandaExact(int(cfg.p), key_freqs, split="all")
    splits = _all_split_losses(_apply(logits, pr, "restricted", cfg), cfg)
    return _record(pr, "restricted", "all", splits, pr.key_freqs, {
        "is_our_variant": True,
        "variant_owner": "GROKVERSE (ours) — full-grid evaluation of the exact operator",
        "same_component_rule_as": "nanda_exact",
        "audit_reference": _AUDIT_REF,
    })


def full_grid_extension_excluded_loss(logits: np.ndarray, key_freqs, cfg) -> dict:
    """OUR variant: the exact excluded rule, read on the FULL (a, b) grid.

    The published excluded loss is a TRAIN-split number (paper §5.1). This
    full-grid reading is GROKVERSE's, kept only so the legacy full-grid numbers
    stay comparable. Never label it a reproduction.
    """
    pr = NandaExact(int(cfg.p), key_freqs, split="all")
    splits = _all_split_losses(_apply(logits, pr, "excluded", cfg), cfg)
    return _record(pr, "excluded", "all", splits, pr.key_freqs, {
        "is_our_variant": True,
        "variant_owner": "GROKVERSE (ours) — full-grid evaluation of the exact operator",
        "same_component_rule_as": "nanda_exact",
        "audit_reference": _AUDIT_REF,
    })


def legacy_broad_mask_variant(logits: np.ndarray, key_freqs, cfg,
                              which: str = "restricted") -> dict:
    """The original outer-product mask, kept runnable — NOT a reproduction.

    Reproduced bit-for-bit (pinned in ``test_core.py`` against the original
    ``progress_measures._mode_indices`` masks) so the old numbers can be
    regenerated and compared. It keeps/deletes cross-frequency blocks, which no
    source ever does, and its restricted and excluded masks are not complements
    of one another (289 kept vs 3360 removed at p=113 with 8 keys). Declared
    split ``all``, the legacy full-grid convention; the other splits are
    reported alongside.
    """
    pr = LegacyBroadMask(int(cfg.p), key_freqs)
    splits = _all_split_losses(_apply(logits, pr, which, cfg), cfg)
    return _record(pr, which, "all", splits, pr.key_freqs, {
        "is_our_variant": True,
        "is_reproduction": False,
        "variant_owner": "GROKVERSE legacy implementation error (audit §3)",
        "why_not_a_reproduction": (
            "keeps/deletes cross-frequency products such as cos(w_18 a)cos(w_15 b); "
            "no source keeps or deletes any k1 != k2 component (audit §1 row 4)"),
        "audit_reference": "docs/MASK_PROTOCOL_AUDIT.md §3",
    })
