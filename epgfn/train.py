"""Trajectory-balance training of the conditional policy.

With P_B = 1 the TB discrepancy per point reduces to
    δ = β_t·log R(x) − ( log Z(c) + log P_F(x | c) ),
(β_t enters because the target is p* ∝ R^{β_t}, i.e. the effective
reward is R^{β_t}); the loss is mean δ² over a batch of (c, x) pairs.

TB is valid off-policy, and X is enumerable, so each batch mixes
on-policy samples with uniform-over-X samples: full support coverage by
construction, keeping O2 a test of amortization rather than exploration.

Held-out conditions come from a deterministic grid; training
draws within an L∞ ball of any held-out feature vector are rejected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .cases import log_reward, target_for
from .conditions import Condition, ConditionRanges
from .policy import ConditionalPolicy
from .target import l1, p_star
from .worlds import World


def default_device() -> str:
    """'cuda' only if a kernel actually runs: torch.cuda.is_available()
    is True on GPUs too old for the installed wheel's kernel set (e.g.
    sm_61 hardware vs sm_75+ builds), and every op then fails."""
    if torch.cuda.is_available():
        try:
            (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception:
            pass
    return "cpu"


@dataclass
class TrainConfig:
    """Hyperparameters and loss selection for `train_policy`/`train_expert`.

    See the inline comments on each field for the exact semantics of
    the loss variants and the pooled-vs-streaming condition sourcing.
    """

    # "tb" (trajectory balance), "exact_kl" (oracle: cross-entropy
    # of the exact full-grid target, −Σ_x p*_c(x)·log p_θ(x|c), mean over
    # the condition batch; needs cond_pool for the target cache; the
    # logz head and the point batch are unused), or "subtb" (SubTB(λ)
    # over all trajectory segments; needs the opt-in flow head (set
    # automatically); a credit-assignment arm).
    loss: str = "tb"
    subtb_lambda: float = 0.9   # geometric segment weight
    steps: int = 2000
    n_conds: int = 8            # conditions per batch
    n_points: int = 64          # points per condition
    uniform_mix: float = 0.5    # share of uniform-over-X points
    lr: float = 1e-3
    lr_logz: float = 1e-2       # ~10x trunk lr, standard for TB
    eval_every: int = 200
    heldout_radius: float = 0.05  # L∞ exclusion in feature space
    # Condition pool: draw this many training conditions once, price the
    # FULL grid's log R for each up front, and train by indexing the
    # (M, N) cache: the exact DRO layer leaves the training loop
    # entirely (the duals are iteration-bound, so full-grid pricing
    # costs ≈ a 64-point subset). None = stream fresh conditions per
    # step with per-batch pricing (the legacy path, kept for the
    # pooled-vs-streaming robustness ablation). Also the fairness
    # device for the exact-KL oracle comparison.
    cond_pool: int | None = 256
    # Training-only clamp on the target logits β_t·log R (None = off).
    # Rationale: mass below e^floor is invisible to the L1 metric
    # (Σ ≈ N·e^floor ≈ 1e-8 at −25, H=32), but ε-region targets
    # (β_t·log ε ≈ −74 under floor/veto) dominate the TB residuals and
    # stall optimization (measured: B/C fail G0 unclamped at 3–5×
    # floor with spiking losses; C passes at 1.8× clamped; A/D, with
    # few ε points, pass either way). Evaluation targets are NEVER
    # clamped; exactness untouched.
    logit_floor: float | None = -25.0
    seed: int = 0
    device: str = "cpu"
    net: dict = field(default_factory=dict)  # ConditionalPolicy kwargs


def _subtb_loss(lp_steps: torch.Tensor, flows: torch.Tensor,
                log_z: torch.Tensor, logr: torch.Tensor,
                lam: float) -> torch.Tensor:
    """SubTB(λ): with F₀ = log Z(c), F_d = β_t·log R (clamped like TB),
    F_t = state flows, and S_t the log-P_F prefix sum, every segment
    i<j contributes λ^{j−i}·((F_i−S_i) − (F_j−S_j))²; normalized by the
    total weight, mean over the batch. TB is the (0,d)-only member."""
    fv = torch.cat([log_z.unsqueeze(1), flows, logr.unsqueeze(1)], dim=1)
    s = torch.cat([torch.zeros_like(log_z).unsqueeze(1),
                   torch.cumsum(lp_steps, dim=1)], dim=1)
    g = fv - s                                       # (B, d+1)
    idx = torch.arange(g.shape[1], device=g.device)
    seg = (idx[None, :] - idx[:, None]).clamp(min=0)
    w = (lam ** seg.float()) * (idx[None, :] > idx[:, None]).float()
    diff = g.unsqueeze(2) - g.unsqueeze(1)           # diff[b,i,j]=G_i−G_j
    return ((w * diff.pow(2)).sum(dim=(1, 2)) / w.sum()).mean()


def ranges_for(world: World,
               ranges: ConditionRanges | None = None) -> ConditionRanges:
    """Bind a `ConditionRanges` to a specific world's case and bounds.

    Args:
        world: World supplying `case`, `beta_bounds`, and `n_outer`.
        ranges: Ranges to bind; a fresh `ConditionRanges()` if omitted.

    Returns:
        The same `ranges` object, mutated in place with the world's
        `case`, `beta_bounds`, and `n_outer`.
    """
    r = ranges or ConditionRanges()
    r.case = world.case
    r.beta_bounds = world.beta_bounds  # per-set bounds
    r.n_outer = world.cfg.n_outer      # D: per-origin ρ axes
    return r


def make_policy_and_opt(world: World, cfg: TrainConfig, dev: str,
                        n_cond_features: int | None = None):
    """Build a ConditionalPolicy sized for `world`/`cfg` and its Adam
    optimizer, with the logZ head at cfg.lr_logz (~10x trunk lr,
    standard for TB) and the rest of the trunk at cfg.lr.
    `n_cond_features`, if given, is the policy's condition-feature
    width (train_expert/w33's arms pass 1 for an unconditional
    policy; train_policy passes ranges.n_features); cfg.net can
    still override it."""
    net = dict(cfg.net)
    if n_cond_features is not None:
        net.setdefault("n_cond_features", n_cond_features)
    net.setdefault("d", world.cfg.d)
    if cfg.loss == "subtb":
        net.setdefault("with_flows", True)
    policy = ConditionalPolicy(world.cfg.H, **net).to(dev)
    logz_params = list(policy.logz_head.parameters())
    logz_ids = {id(p) for p in logz_params}
    trunk = [p for p in policy.parameters() if id(p) not in logz_ids]
    opt = torch.optim.Adam([{"params": trunk, "lr": cfg.lr},
                            {"params": logz_params, "lr": cfg.lr_logz}])
    return policy, opt


def _sample_training_cond(rng, ranges, ho_feats, radius) -> Condition:
    for _ in range(200):
        c = ranges.sample(rng)
        f = ranges.features([c])[0]
        if np.abs(ho_feats - f).max(axis=1).min() >= radius:
            return c
    raise RuntimeError("cannot sample outside held-out exclusion balls; "
                       "shrink heldout_radius or the grid")


def evaluate(policy: ConditionalPolicy, ranges: ConditionRanges,
             heldout: list[Condition], ho_targets: list[np.ndarray],
             device: str = "cpu") -> float:
    """Mean exact L1(policy density, p*_c) over held-out conditions."""
    feats = torch.as_tensor(ranges.features(heldout), dtype=torch.float32,
                            device=device)
    dists = []
    for i, tgt in enumerate(ho_targets):
        lp = policy.log_pf_grid(feats[i]).cpu().numpy().reshape(-1)
        dists.append(l1(np.exp(lp), tgt))
    return float(np.mean(dists))


def train_expert(world: World, cond: Condition, cfg: TrainConfig):
    """Expert: an unconditional policy (n_cond_features=1, constant
    zero features) trained with TB on ONE fixed condition whose log R is
    precomputed once. Returns (policy, history); history rows carry the
    exact L1 to the condition's own target."""
    rng = np.random.default_rng(cfg.seed)
    dev = cfg.device
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    H = world.cfg.H
    shape = (H,) * world.cfg.d

    logr_np = cond.beta_t * log_reward(world, cond)      # (N,) once
    target = p_star(log_reward(world, cond), cond.beta_t)  # exact, unclamped
    if cfg.logit_floor is not None:
        logr_np = np.maximum(logr_np, cfg.logit_floor)

    policy, opt = make_policy_and_opt(world, cfg, dev, n_cond_features=1)

    feats1 = torch.zeros(1, 1, device=dev)
    n_uni = int(round(cfg.n_points * cfg.uniform_mix))
    n_on = cfg.n_points - n_uni
    history = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        uni = rng.integers(0, H ** world.cfg.d, size=n_uni)
        pts = np.stack(np.unravel_index(uni, shape), axis=1)
        if n_on > 0:
            on_pts = policy.sample(feats1.expand(n_on, -1),
                                   gen).cpu().numpy()
            pts = np.concatenate([pts, on_pts], axis=0)
        idx = np.ravel_multi_index(tuple(pts.T), shape)
        points = torch.as_tensor(pts, dtype=torch.long, device=dev)
        logr = torch.as_tensor(logr_np[idx], dtype=torch.float32,
                               device=dev)
        log_pf = policy.log_pf_points(
            points, feats1.expand(cfg.n_points, -1))
        log_z = policy.log_z(feats1)[0]
        delta = logr - log_z - log_pf
        loss = (delta ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            lp = policy.log_pf_grid(feats1[0]).cpu().numpy().reshape(-1)
            history.append({"step": step, "loss": float(loss.item()),
                            "heldout_l1": l1(np.exp(lp), target),
                            "wall_s": time.time() - t0})
    return policy, history


def train_policy(world: World, cfg: TrainConfig,
                 ranges: ConditionRanges | None = None,
                 eval_ranges: ConditionRanges | None = None):
    """Returns (policy, ranges, history); history rows are dicts with
    step / loss / heldout mean L1.

    `eval_ranges`: if given, the held-out grid (evaluation
    conditions AND the training-draw exclusion balls) comes from it,
    while feature normalization uses the TRAINING `ranges` everywhere:
    one normalization source; the eval grid is just conditions."""
    ranges = ranges_for(world, ranges)
    rng = np.random.default_rng(cfg.seed)
    dev = cfg.device
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    H = world.cfg.H
    shape = (H,) * world.cfg.d

    heldout = (eval_ranges or ranges).heldout_grid()
    ho_feats = ranges.features(heldout)  # TRAINING-ranges normalization
    ho_targets = [target_for(world, c) for c in heldout]  # exact, cached

    if cfg.loss not in ("tb", "exact_kl", "subtb"):
        raise ValueError(f"unknown loss {cfg.loss!r}")
    if cfg.loss == "exact_kl" and not cfg.cond_pool:
        raise ValueError("exact_kl needs cond_pool (full-grid target "
                         "cache); streaming would re-price p* per step")

    if cfg.cond_pool:  # per-seed pool: the seed's randomness includes it
        pool = [_sample_training_cond(rng, ranges, ho_feats,
                                      cfg.heldout_radius)
                for _ in range(cfg.cond_pool)]
        pool_feats_np = ranges.features(pool)
        pool_logr = np.stack([log_reward(world, c) for c in pool])
        pool_bt = np.array([c.beta_t for c in pool])
        if cfg.loss == "exact_kl":
            # exact targets, never clamped (the clamp is a TB-residual
            # device; p* mass below e^-25 is ~0 and carries no gradient)
            pool_pstar = torch.as_tensor(
                np.stack([p_star(pool_logr[m], pool_bt[m])
                          for m in range(cfg.cond_pool)]),
                dtype=torch.float32, device=dev)

    policy, opt = make_policy_and_opt(world, cfg, dev,
                                      n_cond_features=ranges.n_features)

    n_uni = int(round(cfg.n_points * cfg.uniform_mix))
    history = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        if cfg.cond_pool:
            sel = rng.integers(0, cfg.cond_pool, size=cfg.n_conds)
            feats = torch.as_tensor(pool_feats_np[sel],
                                    dtype=torch.float32, device=dev)
        else:
            conds = [_sample_training_cond(rng, ranges, ho_feats,
                                           cfg.heldout_radius)
                     for _ in range(cfg.n_conds)]
            feats = torch.as_tensor(ranges.features(conds),
                                    dtype=torch.float32, device=dev)
        if cfg.loss == "exact_kl":
            logp_grid = policy.log_pf_grid_train(feats)
            logp_grid = logp_grid.reshape(cfg.n_conds, -1)
            loss = -(pool_pstar[sel] * logp_grid).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % cfg.eval_every == 0 or step == cfg.steps:
                mean_l1 = evaluate(policy, ranges, heldout, ho_targets,
                                   dev)
                history.append({"step": step, "loss": float(loss.item()),
                                "heldout_l1": mean_l1,
                                "wall_s": time.time() - t0})
            continue
        pts_list, logr_list, rows = [], [], []
        for i in range(cfg.n_conds):
            uni = rng.integers(0, H ** world.cfg.d, size=n_uni)
            uni_pts = np.stack(np.unravel_index(uni, shape), axis=1)
            n_on = cfg.n_points - n_uni
            if n_on > 0:
                cf = feats[i].unsqueeze(0).expand(n_on, -1)
                on_pts = policy.sample(cf, gen).cpu().numpy()
                pts = np.concatenate([uni_pts, on_pts], axis=0)
            else:
                pts = uni_pts
            idx = np.ravel_multi_index(tuple(pts.T), shape)
            if cfg.cond_pool:
                m = sel[i]
                lr_i = pool_bt[m] * pool_logr[m, idx]
            else:
                c = conds[i]
                lr_i = c.beta_t * log_reward(world, c, idx)
            if cfg.logit_floor is not None:
                lr_i = np.maximum(lr_i, cfg.logit_floor)
            logr_list.append(lr_i)
            pts_list.append(pts)
            rows.append(np.full(cfg.n_points, i))
        points = torch.as_tensor(np.concatenate(pts_list),
                                 dtype=torch.long, device=dev)
        cond_rows = torch.as_tensor(np.concatenate(rows),
                                    dtype=torch.long, device=dev)
        logr = torch.as_tensor(np.concatenate(logr_list),
                               dtype=torch.float32, device=dev)

        batch_feats = feats[cond_rows]
        log_z = policy.log_z(feats)[cond_rows]
        if cfg.loss == "subtb":
            lp_steps, flows = policy.log_pf_points_flows(points,
                                                         batch_feats)
            loss = _subtb_loss(lp_steps, flows, log_z, logr,
                               cfg.subtb_lambda)
        else:
            log_pf = policy.log_pf_points(points, batch_feats)
            delta = logr - log_z - log_pf
            loss = (delta ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            mean_l1 = evaluate(policy, ranges, heldout, ho_targets, dev)
            history.append({"step": step, "loss": float(loss.item()),
                            "heldout_l1": mean_l1,
                            "wall_s": time.time() - t0})
    return policy, ranges, history
