"""Condition vector c = (β_t, w, risk), its
sampling ranges, and the fixed preprocessing that feeds the policy network.

Per-origin risk: every weighted origin carries its own
(β, ρ); the thresholded origin in Case C carries a veto margin δ; Case D
conditions its O inner radii directly (aimed distrust) plus the outer
tail level β_out (the joint-satisfaction dial). The former
single-pair condition embeds in each family as the tied member
(`tied_risk`).

w = (w_g, w_s) is constrained to the simplex (w_s = 1 − w_g). Preprocessing
warps each axis to roughly [−1, 1] over its declared range so no capacity
is spent learning scales: β_t is log-scaled (it multiplies log R), ρ and s
are log1p-scaled (nonnegative, wide), each β axis is normalized on its
per-set valid interval [bound, 1], δ is linear.

Feature layout is [β_t, w_g, *risk-axes], so cases differ only in the
risk block: A (β, ρ) → 4 features; B (β⁺, ρ⁺, β⁻, ρ⁻) → 6;
C (β⁺, ρ⁺, δ) → 5; D (β_in, ρ_1..ρ_O, β_out) → 4 + O (8 at O = 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RiskA:
    """Single origin S⁰. A
    can carry a guard (WorldConfig.k_guard > 0): `delta` is then the
    C-style veto margin (veto at a_d ≥ c_d − δ − σ); on guard-free
    worlds it is inert. 0.0 = identity."""
    beta: float
    rho: float
    sigma: float = 0.0  # score-robustness margin; 0 = off
    geometry: str = "kl"  # ambiguity-ball family; arm label, not
    #                       a network feature; radii are ball-specific
    delta: float = 0.0  # guard veto margin; inert without a guard


@dataclass(frozen=True)
class RiskB:
    """Two origins: S⁺ (p, lower tail) and S⁻ (q, upper tail)."""
    beta_p: float
    rho_p: float
    beta_m: float
    rho_m: float
    sigma: float = 0.0
    geometry: str = "kl"  # applied to both sides' balls


@dataclass(frozen=True)
class RiskC:
    """Weighted origin S⁺ plus thresholded origin D: veto margin δ
    (veto fires at a_d ≥ c_d − δ)."""
    beta: float
    rho: float
    delta: float = 0.0
    sigma: float = 0.0
    geometry: str = "kl"


@dataclass(frozen=True)
class RiskD:
    """O inner origins with per-origin radii ρ_o (aimed distrust) plus
    the conditioned outer tail level β_out, the joint-satisfaction dial
    (β_out → 1 compensatory, β_out → its π bound conjunctive).
    `rho` is a tuple of O floats or a scalar (shared radii = the tied
    member). The outer radius ρ_out is a conditioned axis: the ball
    centre is the model-implied π, mechanically identical to the
    inner ρ_o. The identity default None resolves to the world
    constant `WorldConfig.rho_out`, leaving that path unchanged when
    the axis is unused."""
    beta_in: float
    rho: object  # float (shared/tied) or tuple of O floats
    beta_out: float = 0.5
    sigma: float = 0.0
    rho_out: float | None = None  # None → cfg.rho_out (identity default)
    geometry: str = "kl"  # applied to BOTH levels' balls


def tied_risk(case: str, beta: float, rho: float, delta: float = 0.0):
    """The shared-(β, ρ) member of each case's risk family: the
    original single-pair condition embeds in the new family through
    this. For D: shared radii at the reference outer level
    (β_out = 0.5, the original fixed value)."""
    if case == "A":
        return RiskA(beta, rho, delta=delta)  # δ inert without a guard
    if case == "B":
        return RiskB(beta, rho, beta, rho)
    if case == "C":
        return RiskC(beta, rho, delta)
    if case == "D":
        return RiskD(beta, rho, 0.5)
    raise ValueError(f"unknown case {case!r}")


@dataclass(frozen=True)
class Condition:
    """A full condition vector: the outer tail level, the objective
    weight split, and the case-specific risk block.

    Attributes:
        beta_t: Outer tail level; the target is p* ∝ R^beta_t.
        w_g: Weight on the auxiliary objective g(x); w_s = 1 - w_g.
        risk: One of RiskA/RiskB/RiskC/RiskD, per `Condition.risk`'s
            owning case.
    """

    beta_t: float
    w_g: float
    risk: object  # RiskA | RiskB | RiskC | RiskD

    @property
    def w_s(self) -> float:
        """Weight on the score objective (1 - w_g)."""
        return 1.0 - self.w_g


_DEFAULT_BOUNDS = {"A": {"n": 0.05}, "B": {"p": 0.05, "m": 0.05},
                   "C": {"p": 0.05}, "D": {"in": 0.05, "out": 0.05}}


@dataclass
class ConditionRanges:
    """Sampling ranges for one case's condition space, plus the
    per-world bounds and feature-warp bookkeeping needed to turn a
    `Condition` into network features.

    Populated in two stages: the dataclass defaults declare the
    case-independent axis ranges; `train.ranges_for` then fills in
    `beta_bounds`/`n_outer` from a specific `World` before `sample`,
    `heldout_grid`, or `features` are called.
    """

    case: str = "A"
    beta_t: tuple = (0.5, 8.0)      # log-uniform
    w_g: tuple = (0.1, 0.9)         # uniform
    rho: tuple = (0.0, 1.6)         # per ρ axis
    delta: tuple = (0.0, 0.1)       # C: veto margin range
    n_outer: int = 4                # D: number of inner origins O
    # B only: sample ONE β for both sides (intrinsic dim 5;
    # features stay 6-D with the value duplicated; capacity identical).
    # The shared β respects BOTH per-set bounds (max of the two).
    tie_beta: bool = False
    # per-key upper ends for the β axes (default 1.0); set by
    # shrink(): the lower ends are validity bounds and never move.
    beta_tops: dict = field(default_factory=dict)
    # per-set lower bounds for the β axes;
    # overwritten from the world (train.ranges_for). Keys per case:
    # A {"n"}, B {"p","m"}, C {"p"}, D {"in","out"}: "out" is the
    # outer tail level's bound (min π), the conjunctive end of the
    # joint-satisfaction dial.
    beta_bounds: dict = field(default_factory=dict)
    # Score-robustness margin σ as a condition axis. OFF by
    # default, untouched unless enabled (σ = 0 on all blocks).
    # When on: +1 feature (LINEAR warp), σ ~ U(range),
    # held-out grid gains a 3-point σ axis (2 for D, like its other
    # risk axes).
    use_sigma: bool = False
    sigma: tuple = (0.0, 0.2)
    # Conditioned outer radius ρ_out (D only). OFF by default,
    # untouched unless enabled (rho_out=None on all blocks,
    # resolving to the world constant). When on: +1 feature (log1p
    # warp, SAME declared ρ range as the other radii), ρ_out ~ U(rho),
    # held-out grid gains a 2-point ρ_out axis (like D's other risk
    # axes). σ stays the LAST feature.
    use_rho_out: bool = False
    # guarded-A veto margin δ as a condition axis (A only; the
    # world must carry a guard, k_guard > 0). OFF by default, untouched
    # unless enabled (delta=0.0 on all A blocks). When on:
    # +1 feature (LINEAR warp, C's declared δ range), 3-point
    # held-out axis; σ stays the LAST feature.
    use_guard_delta: bool = False

    @property
    def n_features(self) -> int:
        """Network feature-vector width for this case and its
        optionally-enabled axes (use_rho_out/use_guard_delta/use_sigma)."""
        base = (4 + self.n_outer if self.case == "D"
                else {"A": 4, "B": 6, "C": 5}[self.case])
        return (base + (1 if self.use_rho_out else 0)
                + (1 if self.use_guard_delta else 0)
                + (1 if self.use_sigma else 0))

    def _check_rho_out(self) -> None:
        if self.use_rho_out and self.case != "D":
            raise ValueError("use_rho_out is defined for case D only")
        if self.use_guard_delta and self.case != "A":
            raise ValueError("use_guard_delta is defined for case A "
                             "only (C always conditions its δ)")

    def _b(self, key: str) -> tuple:
        lo = self.beta_bounds.get(key, _DEFAULT_BOUNDS[self.case][key])
        return (lo, self.beta_tops.get(key, 1.0))

    def _b_tied(self) -> tuple:
        """Valid interval for a β shared across B's two sides: the max
        of the per-set lower bounds (mirrors `world.beta_min`)."""
        if self.case != "B" or not self.tie_beta:
            raise ValueError("tie_beta is defined for case B only")
        return (max(self._b("p")[0], self._b("m")[0]), 1.0)

    def shrink(self, factor: float = 0.8) -> "ConditionRanges":
        """A copy with every axis width multiplied by `factor`.
        Free axes (β_t in LOG space, w_g) shrink symmetrically toward
        their midpoint; validity-pinned axes keep their lower end and
        move only the top (β axes: the lower bound is a validity
        constraint; ρ and δ: their lower end is 0). Call AFTER
        `ranges_for` has set the per-world bounds. Extrapolation test
        downstream: a condition is `extrap` iff any feature computed
        with the SHRUNK (training) ranges falls outside [−1, 1]."""
        import copy

        r = copy.deepcopy(self)

        def sym(lo, hi):
            m, h = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
            return (m - h, m + h)

        llo, lhi = sym(np.log(self.beta_t[0]), np.log(self.beta_t[1]))
        r.beta_t = (float(np.exp(llo)), float(np.exp(lhi)))
        r.w_g = tuple(float(v) for v in sym(*self.w_g))

        def top_only(lohi):
            lo, hi = lohi
            return (float(lo), float(lo + factor * (hi - lo)))

        r.rho = top_only(self.rho)
        r.delta = top_only(self.delta)
        keys = {"A": ["n"], "B": ["p", "m"],
                "C": ["p"], "D": ["in", "out"]}[self.case]
        r.beta_tops = {k: top_only(self._b(k))[1] for k in keys}
        return r

    def sample(self, rng: np.random.Generator) -> Condition:
        """Draw one random `Condition` from these ranges.

        Args:
            rng: NumPy random generator used for every draw.

        Returns:
            A `Condition` whose risk block matches `self.case`.

        Raises:
            ValueError: If `self.case` is not one of "A"/"B"/"C"/"D",
                if `tie_beta` is set on a case other than "B", or if
                `use_rho_out`/`use_guard_delta` is set on a case that
                does not define that axis.
        """
        def u(lohi):
            return float(rng.uniform(*lohi))

        bt = float(np.exp(rng.uniform(np.log(self.beta_t[0]),
                                      np.log(self.beta_t[1]))))
        wg = u(self.w_g)
        if self.tie_beta and self.case != "B":
            raise ValueError("tie_beta is defined for case B only")
        if self.case == "A":
            risk = RiskA(u(self._b("n")), u(self.rho))
        elif self.case == "B":
            if self.tie_beta:
                b = u(self._b_tied())
                risk = RiskB(b, u(self.rho), b, u(self.rho))
            else:
                risk = RiskB(u(self._b("p")), u(self.rho),
                             u(self._b("m")), u(self.rho))
        elif self.case == "C":
            risk = RiskC(u(self._b("p")), u(self.rho), u(self.delta))
        elif self.case == "D":
            risk = RiskD(u(self._b("in")),
                         tuple(u(self.rho) for _ in range(self.n_outer)),
                         u(self._b("out")))
        else:
            raise ValueError(f"unknown case {self.case!r}")
        self._check_rho_out()
        if self.use_rho_out:
            import dataclasses
            risk = dataclasses.replace(risk, rho_out=u(self.rho))
        if self.use_guard_delta:
            import dataclasses
            risk = dataclasses.replace(risk, delta=u(self.delta))
        if self.use_sigma:
            import dataclasses
            risk = dataclasses.replace(risk, sigma=u(self.sigma))
        return Condition(bt, wg, risk)

    def heldout_grid(self, n_per_axis: int = 3) -> list[Condition]:
        """Deterministic grid for O2 evaluation, w_g fixed mid-range;
        excluded from training by construction (training samples are
        continuous draws, and the trainer additionally rejects draws
        within a ball of these; see train.py). D uses 2 points per risk
        axis (its risk block has 2 + O axes). Sizes at n=3, O=4:
        A 27, B 243 (81 with tie_beta), C 81, D 192."""
        import itertools

        n = n_per_axis
        bts = np.exp(np.linspace(np.log(self.beta_t[0]),
                                 np.log(self.beta_t[1]), n))
        wg = 0.5 * (self.w_g[0] + self.w_g[1])

        def lin(lohi, k=n):
            return np.linspace(lohi[0], lohi[1], k)

        rhos = lin(self.rho)
        out = []
        if self.tie_beta and self.case != "B":
            raise ValueError("tie_beta is defined for case B only")
        if self.case == "A":
            blocks = [RiskA(float(b), float(r))
                      for b in lin(self._b("n")) for r in rhos]
        elif self.case == "B":
            if self.tie_beta:  # tied family: 3β×3ρ⁺×3ρ⁻ (81 with β_t)
                blocks = [RiskB(float(b), float(rp), float(b), float(rm))
                          for b in lin(self._b_tied()) for rp in rhos
                          for rm in rhos]
            else:
                blocks = [RiskB(float(bp), float(rp), float(bm), float(rm))
                          for bp in lin(self._b("p")) for rp in rhos
                          for bm in lin(self._b("m")) for rm in rhos]
        elif self.case == "C":
            blocks = [RiskC(float(b), float(r), float(d))
                      for b in lin(self._b("p")) for r in rhos
                      for d in lin(self.delta)]
        elif self.case == "D":
            rho2 = lin(self.rho, 2)
            blocks = [RiskD(float(b), tuple(float(r) for r in rv),
                            float(bo))
                      for b in lin(self._b("in"), 2)
                      for rv in itertools.product(rho2,
                                                  repeat=self.n_outer)
                      for bo in lin(self._b("out"), 2)]
        else:
            raise ValueError(f"unknown case {self.case!r}")
        self._check_rho_out()
        if self.use_rho_out:
            import dataclasses
            blocks = [dataclasses.replace(blk, rho_out=float(ro))
                      for blk in blocks for ro in lin(self.rho, 2)]
        if self.use_guard_delta:
            import dataclasses
            blocks = [dataclasses.replace(blk, delta=float(dl))
                      for blk in blocks for dl in lin(self.delta)]
        if self.use_sigma:
            import dataclasses
            sig = lin(self.sigma, 2 if self.case == "D" else n)
            blocks = [dataclasses.replace(blk, sigma=float(s))
                      for blk in blocks for s in sig]
        for bt in bts:
            out.extend(Condition(float(bt), wg, blk) for blk in blocks)
        return out

    def features(self, conds: list[Condition]) -> np.ndarray:
        """(n, n_features) network features in ~[−1, 1]."""
        self._check_rho_out()
        out = np.empty((len(conds), self.n_features))
        lr0, lr1 = np.log1p(self.rho[0]), np.log1p(self.rho[1])
        for i, c in enumerate(conds):
            out[i, 0] = _unit(np.log(c.beta_t),
                              np.log(self.beta_t[0]), np.log(self.beta_t[1]))
            out[i, 1] = _unit(c.w_g, *self.w_g)
            r = c.risk
            if self.case == "A":
                out[i, 2] = _unit(r.beta, *self._b("n"))
                out[i, 3] = _unit(np.log1p(r.rho), lr0, lr1)
                if self.use_guard_delta:  # σ stays last
                    out[i, 4] = _unit(getattr(r, "delta", 0.0),
                                      *self.delta)
            elif self.case == "B":
                out[i, 2] = _unit(r.beta_p, *self._b("p"))
                out[i, 3] = _unit(np.log1p(r.rho_p), lr0, lr1)
                out[i, 4] = _unit(r.beta_m, *self._b("m"))
                out[i, 5] = _unit(np.log1p(r.rho_m), lr0, lr1)
            elif self.case == "C":
                out[i, 2] = _unit(r.beta, *self._b("p"))
                out[i, 3] = _unit(np.log1p(r.rho), lr0, lr1)
                out[i, 4] = _unit(r.delta, *self.delta)
            elif self.case == "D":
                out[i, 2] = _unit(r.beta_in, *self._b("in"))
                rho_vec = np.broadcast_to(
                    np.asarray(r.rho, dtype=float), (self.n_outer,))
                for o in range(self.n_outer):
                    out[i, 3 + o] = _unit(np.log1p(rho_vec[o]), lr0, lr1)
                out[i, 3 + self.n_outer] = _unit(r.beta_out,
                                                 *self._b("out"))
            if self.use_rho_out:
                self._check_rho_out()
                # a None on the block would silently encode the world
                # constant as 0; reject instead
                ro = getattr(r, "rho_out", None)
                if ro is None:
                    raise ValueError("use_rho_out=True but the condition "
                                     "carries rho_out=None")
                out[i, 4 + self.n_outer] = _unit(np.log1p(ro), lr0, lr1)
            if self.use_sigma:
                out[i, -1] = _unit(getattr(r, "sigma", 0.0),
                                   *self.sigma)
        return out


def expert_subset(ranges: ConditionRanges,
                  n_per_axis: int = 3) -> list[Condition]:
    """Deterministic expert subset: members of `heldout_grid()`
    varying β_t over its {first, mid, last} grid indices × the FIRST
    risk axis (A/C β, B β⁺, D β_in) over its {first, mid, last}; every
    other axis at its mid index ((n−1)//2; for 2-point axes that is 0,
    so D's 2-point first axis yields 2 distinct values → K = 6 there,
    K = 9 elsewhere). Pure function of the ranges object."""
    n = n_per_axis
    bts = np.exp(np.linspace(np.log(ranges.beta_t[0]),
                             np.log(ranges.beta_t[1]), n))
    wg = 0.5 * (ranges.w_g[0] + ranges.w_g[1])

    def lin(lohi, k=n):
        return np.linspace(lohi[0], lohi[1], k)

    def ends_mid(arr):
        idx = sorted({0, (len(arr) - 1) // 2, len(arr) - 1})
        return [float(arr[i]) for i in idx]

    def mid(arr):
        return float(arr[(len(arr) - 1) // 2])

    rho_mid = mid(lin(ranges.rho))
    case = ranges.case
    if case == "A":
        # guarded-A δ sits at its mid grid point, like C's
        d = mid(lin(ranges.delta)) if ranges.use_guard_delta else 0.0
        blocks = [RiskA(b, rho_mid, delta=d)
                  for b in ends_mid(lin(ranges._b("n")))]
    elif case == "B":
        bm = mid(lin(ranges._b("m")))
        blocks = [RiskB(bp, rho_mid, bm, rho_mid)
                  for bp in ends_mid(lin(ranges._b("p")))]
    elif case == "C":
        d = mid(lin(ranges.delta))
        blocks = [RiskC(b, rho_mid, d)
                  for b in ends_mid(lin(ranges._b("p")))]
    elif case == "D":
        rho2_mid = mid(lin(ranges.rho, 2))
        bo = mid(lin(ranges._b("out"), 2))
        # under use_rho_out every axis sits at its mid index, so
        # ρ_out takes the 2-point axis' mid (= its first point, like
        # D's other 2-point axes); None would crash features()
        ro = rho2_mid if ranges.use_rho_out else None
        blocks = [RiskD(b, tuple(rho2_mid for _ in range(ranges.n_outer)),
                        bo, rho_out=ro)
                  for b in ends_mid(lin(ranges._b("in"), 2))]
    else:
        raise ValueError(f"unknown case {case!r}")
    return [Condition(bt, wg, blk)
            for bt in ends_mid(bts) for blk in blocks]


def _unit(v: float, lo: float, hi: float) -> float:
    return 2.0 * (v - lo) / (hi - lo) - 1.0
