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
=========================  =========  ===========  ===================================
name                       kept       removed      what it assumes
=========================  =========  ===========  ===================================
``legacy_broad_mask``      289        3360 (26.3%) every pair of key rows, cross-freq
``same_frequency_block``   33         32           same frequency on both axes
``sum_directions_only``    17         16           only cos/sin(w_k(a+b))
``nanda_exact``            —          —            NOT IMPLEMENTED (see below)
=========================  =========  ===========  ===================================

``nanda_exact`` deliberately raises. Reconstructing the published protocol
requires reading arXiv:2301.05217 and the authors' released notebook (WP-0/WP-1)
— which axes are transformed, whether cross-frequency blocks are permitted, how
the constant term and the cos/sin pairing are handled, and which split each loss
is evaluated on. Guessing it and calling it "nanda_exact" would manufacture
exactly the false reproduction claim this module exists to remove.

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


class NandaExact(Protocol):
    """Placeholder for the protocol as published — NOT reconstructed yet."""

    name = "nanda_exact"

    def _unimplemented(self, *_a, **_k):
        raise NotImplementedError(
            "nanda_exact is not implemented. It must be reconstructed from the "
            "primary source (arXiv:2301.05217 + the authors' released notebook) "
            "in WP-0/WP-1 — which axes are transformed, whether cross-frequency "
            "blocks are permitted, how the constant and cos/sin pairing are "
            "handled, and which split each loss is evaluated on. Guessing it "
            "would manufacture the false reproduction claim mask_protocols.py "
            "exists to remove. Use same_frequency_block or sum_directions_only "
            "and label them as OUR variants.")

    restrict = _unimplemented
    exclude = _unimplemented
    n_kept = property(_unimplemented)
    n_removed = property(_unimplemented)


_REGISTRY: dict[str, type[Protocol]] = {
    LegacyBroadMask.name: LegacyBroadMask,
    SameFrequencyBlock.name: SameFrequencyBlock,
    SumDirectionsOnly.name: SumDirectionsOnly,
    NandaExact.name: NandaExact,
}

#: Variants that can actually be computed today, tightest-hypothesis last.
IMPLEMENTED: tuple[str, ...] = (
    LegacyBroadMask.name, SameFrequencyBlock.name, SumDirectionsOnly.name,
)

ALL_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def build_protocol(name: str, p: int, key_freqs) -> Protocol:
    if name not in _REGISTRY:
        raise ValueError(f"unknown protocol {name!r}; choices: {list(_REGISTRY)}")
    return _REGISTRY[name](p, key_freqs)


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
