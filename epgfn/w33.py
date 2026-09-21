"""Reproduction arms.

Native regime: an UNCONDITIONAL student policy trained with TB on ONE
fixed condition of a (typically sequence, high-sparsity) world: the
published adaptive-teacher setting. The arms share the net, budget,
seed, and batch size and differ only in where the off-policy half of
each batch comes from (and, for "contrastive", one auxiliary loss):

- "onpolicy": no off-policy half (mix = 0 baseline).
- "mix":      uniform-over-X half (the baseline comparator).
- "replay":   reward-prioritized replay: visited states, sampled with
              probability ∝ exp(clamped β_t·log R).
- "teacher":  a second policy trained concurrently with TB on a reward
              built from the student's TB discrepancy over the teacher's
              own samples (Kim et al., adaptive teachers for amortized
              samplers), the student's off-policy half coming from it.
              Their eq. (4) is the bare squared discrepancy; eq. (5)
              multiplies it by (1 + C·1[δ > 0]) with C = 19 so the
              teacher favours states the student UNDERsamples rather
              than chasing large discrepancies of either sign; eq. (6)
              optionally mixes in α·log R. We use C = 19 and α = 0,
              the paper's own setting for exploration-intensive tasks.
- "contrastive": two-buffer contrastive replay (the production D⁺/D⁻
              mechanism, battery-native form). Every visited state is
              classified by the ε-criterion (log R > log ε at the fixed
              condition): positives feed D⁺ (unique states, capacity
              `buf_cap`, lowest-reward evicted; the buffer accumulates
              the best found), negatives feed D⁻ (FIFO over recent
              visits; it tracks the CURRENT policy's failures). The
              off-policy half of the TB batch is drawn from D⁺; the
              loss adds α·L_aux, an InfoNCE term over balanced
              B⁺/B⁻ draws with s(τ) = log P_F (states are terminal, so
              trajectory log-prob is the point log-prob). L_aux is
              self-limiting: once every buffered negative has
              negligible probability the log-ratio vanishes. In a
              world whose ε-set is empty at the fixed condition
              (template A without a guard) D⁻ stays empty, L_aux is
              inert, and the arm reduces to top-R positive replay
              (reported by loss_aux = nan, not hidden).

Native metrics: the mode set is `satisfaction_mask` at
`challenge_level(world, min_frac=0.01)`: exact, world-level machinery.
Two families are reported. Training discovery counts a mode as found
once it appears in ANY training batch (curves of cumulative discovery
vs samples). Policy coverage is a property of the final trained
sampler alone: the expected fraction of the mode set a fresh budget of
draws from it would find, computed exactly from the terminating
distribution (no sampling noise) and read against the same quantity
for the exact target, which is the true ceiling at that budget on a
sparse world. The edge-share metric (share of batch points in the
Hamming-1 shell of the ε-set) tests whether training oversamples the
boundary of the ε-set.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .cases import EPS_REWARD, log_reward, log_reward_graded
from .conditions import Condition, tied_risk
from .o3 import challenge_level, satisfaction_mask
from .policy import ConditionalPolicy
from .target import l1, p_star
from .train import TrainConfig, make_policy_and_opt
from .worlds import World

ARMS = ("onpolicy", "mix", "replay", "teacher", "contrastive")


def fixed_condition(world: World) -> Condition:
    """The arm-shared target condition: the hardness probe-on member
    (risk-on, mid w_g), matching the gate's convention so every world
    is guaranteed non-degenerate at it."""
    lo = world.beta_min
    beta = 0.25 if 0.25 >= lo else min(0.99, lo * 1.05)
    return Condition(4.0, 0.3, tied_risk(world.case, beta, 0.5))


def mode_indices(world: World, min_frac: float = 0.01) -> np.ndarray:
    """The mode set: satisfying states at the world's challenge
    level (the hardest non-degenerate joint requirement)."""
    t_star = challenge_level(world, min_frac=min_frac)
    return np.flatnonzero(satisfaction_mask(world, t_star))


def expected_coverage(p: np.ndarray, is_mode: np.ndarray,
                      n_draws: int) -> float:
    """Expected fraction of the mode set discovered in `n_draws` i.i.d.
    draws from `p`: mean over modes of 1 - (1 - p_x)^n.

    This is what coverage means for a TRAINED SAMPLER, as opposed to
    what a training run happened to visit. Read it against the same
    quantity computed for the exact target: on a sparse world even a
    perfect sampler does not reach 1 at a finite budget, so 1 is the
    wrong reference.
    """
    if n_draws <= 0:
        return 0.0
    pm = np.asarray(p, dtype=float)[is_mode]
    if pm.size == 0:
        return float("nan")
    # 1 - (1-p)^n via log1p, stable when p is tiny
    miss = np.exp(n_draws * np.log1p(-np.clip(pm, 0.0, 1.0 - 1e-15)))
    return float(np.mean(1.0 - miss))


def eps_shell(world: World, cond: Condition) -> np.ndarray:
    """Boolean (N,): states on the Hamming-1 boundary of the ε-set
    (R == ε). A state is in the shell iff its coordinate line along
    some axis contains both ε and non-ε states (covers both sides of
    the boundary)."""
    H, d = world.cfg.H, world.cfg.d
    eps_mask = np.isclose(log_reward(world, cond),
                          np.log(EPS_REWARD)).reshape((H,) * d)
    shell = np.zeros_like(eps_mask)
    for t in range(d):
        mixed = (eps_mask.any(axis=t, keepdims=True)
                 & ~eps_mask.all(axis=t, keepdims=True))
        shell |= np.broadcast_to(mixed, eps_mask.shape)
    return shell.reshape(-1)


def infonce(lp_pos: torch.Tensor, lp_neg: torch.Tensor) -> torch.Tensor:
    """L_aux: mean over positives of
    −log[ exp(s⁺) / (exp(s⁺) + Σ_{B⁻} exp(s⁻)) ],
    each positive contrasted against the WHOLE negative batch.
    Computed via logsumexp so buffered negatives the
    policy already suppresses (s⁻ ≈ −40) underflow to exactly zero
    contribution: the self-limiting property is numerically real."""
    denom = torch.logsumexp(
        torch.cat([lp_pos.unsqueeze(1),
                   lp_neg.unsqueeze(0).expand(len(lp_pos), -1)], dim=1),
        dim=1)
    return (denom - lp_pos).mean()


def run_arm(world: World, cond: Condition, cfg: TrainConfig,
            arm: str, alpha_aux: float = 1.0, buf_cap: int = 1024,
            graded_kappa: float = 0.0, teacher_c: float = 19.0,
            teacher_alpha: float = 0.0, teacher_eps: float = 1e-8
            ) -> tuple[ConditionalPolicy, list[dict]]:
    """Train one arm; returns (student, history). History rows carry
    both discovery families described in the module docstring (see
    each key's comment below), plus loss / samples / edge_share
    (cumulative) / l1 (exact, to the unclamped target) / wall_s; the
    contrastive arm adds loss_aux (nan until both buffers are
    non-empty). `alpha_aux` and `buf_cap` are contrastive-only knobs
    (α on L_aux; capacity of EACH buffer). `graded_kappa` > 0 replaces
    the flat epsilon on excluded points with epsilon * exp(-kappa *
    violation depth), removing the plateau while leaving the
    constraint boundary where it was; the target moves with the
    reward, as it must, since this is a different reward, not a
    different evaluation of the same one. `teacher_c`/`teacher_alpha`/
    `teacher_eps` are the teacher-arm eq. (5)/(6) knobs; other arms
    ignore all four. `loss` in the history is always the TB term
    alone, so it stays comparable across arms."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    rng = np.random.default_rng(cfg.seed)
    dev = cfg.device
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    H, d = world.cfg.H, world.cfg.d
    n = world.n_points

    _lr = (log_reward(world, cond) if graded_kappa == 0.0
           else log_reward_graded(world, cond, graded_kappa))
    logr_np = cond.beta_t * _lr
    target = p_star(_lr, cond.beta_t)
    logr_tr = (np.maximum(logr_np, cfg.logit_floor)
               if cfg.logit_floor is not None else logr_np)

    modes = mode_indices(world)
    is_mode = np.zeros(n, dtype=bool)
    is_mode[modes] = True
    # LIVE modes: satisfying states the reward does not clamp to
    # epsilon. Satisfaction is a property of the scores; the reward is
    # a property of the composition. Only case B can disagree, since
    # only case B subtracts (Φ(promoted) − γΦ(suppressed)), so a state
    # can clear the score threshold while the composed reward turns
    # negative and clamps; A, C and D only aggregate, so every
    # satisfying state of theirs is live and this restriction is the
    # identity there. It matters because coverage of a clamped mode is
    # unreachable by construction: the exact target gives it
    # epsilon^beta_t mass, so no sampler finds it and the ceiling, not
    # just the policy, falls.
    is_live = is_mode & (_lr > np.log(EPS_REWARD) + 1e-12)
    shell = eps_shell(world, cond)

    student, opt = make_policy_and_opt(world, cfg, dev, n_cond_features=1)
    teacher = t_opt = None
    if arm == "teacher":
        teacher, t_opt = make_policy_and_opt(world, cfg, dev,
                                             n_cond_features=1)

    # replay buffer: priority per unique visited state (∝ exp(logr_tr))
    buf_pri = np.zeros(n)
    # contrastive buffers: D⁺ unique states w/ lowest-R eviction, D⁻
    # FIFO over recent visits (duplicates across steps deliberate; the
    # negative batch weights the policy's CURRENT failure frequency).
    # Classification is the ε-criterion on the UNCLAMPED reward at the
    # fixed condition: the same boundary eps_shell measures.
    dead = np.isclose(log_reward(world, cond), np.log(EPS_REWARD))
    d_pos = np.zeros(0, dtype=np.int64)
    d_neg = np.zeros(0, dtype=np.int64)

    feats1 = torch.zeros(1, 1, device=dev)
    n_off = 0 if arm == "onpolicy" else cfg.n_points // 2
    seen = np.zeros(n, dtype=bool)
    shell_hits = 0
    samples = 0
    history: list[dict] = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        # --- assemble the batch ---------------------------------
        if arm == "mix" and n_off:
            off_idx = rng.integers(0, n, size=n_off)
        elif arm == "replay" and n_off and buf_pri.sum() > 0:
            p = buf_pri / buf_pri.sum()
            off_idx = rng.choice(n, size=n_off, p=p)
        elif arm == "teacher" and n_off:
            off_pts = teacher.sample(feats1.expand(n_off, -1),
                                     gen).cpu().numpy()
            off_idx = np.ravel_multi_index(tuple(off_pts.T), (H,) * d)
        elif arm == "contrastive" and n_off and len(d_pos):
            # TB half from D⁺ (with replacement: the buffer may still
            # be smaller than the half-batch early on)
            off_idx = rng.choice(d_pos, size=n_off)
        else:  # onpolicy, or replay/contrastive before first buffering
            off_idx = np.zeros(0, dtype=np.int64)
        n_fill = cfg.n_points - len(off_idx)
        on_pts = student.sample(feats1.expand(n_fill, -1),
                                gen).cpu().numpy()
        on_idx = np.ravel_multi_index(tuple(on_pts.T), (H,) * d)
        idx = np.concatenate([off_idx, on_idx])
        pts = np.stack(np.unravel_index(idx, (H,) * d), axis=1)
        points = torch.as_tensor(pts, dtype=torch.long, device=dev)

        # --- student TB update ----------------------------------
        logr = torch.as_tensor(logr_tr[idx], dtype=torch.float32,
                               device=dev)
        log_pf = student.log_pf_points(
            points, feats1.expand(len(idx), -1))
        log_z = student.log_z(feats1)[0]
        delta = logr - log_z - log_pf
        loss = (delta ** 2).mean()
        total, loss_aux = loss, float("nan")
        if arm == "contrastive" and len(d_pos) and len(d_neg):
            # balanced |B⁺| = |B⁻| = n_off, uniform over each buffer
            pairs = np.stack(np.unravel_index(
                np.concatenate([rng.choice(d_pos, size=n_off),
                                rng.choice(d_neg, size=n_off)]),
                (H,) * d), axis=1)
            lp = student.log_pf_points(
                torch.as_tensor(pairs, dtype=torch.long, device=dev),
                feats1.expand(2 * n_off, -1))
            aux = infonce(lp[:n_off], lp[n_off:])
            total = loss + alpha_aux * aux
            loss_aux = float(aux.item())
        opt.zero_grad()
        total.backward()
        opt.step()

        # --- arm bookkeeping -------------------------------------
        if arm == "replay":
            buf_pri[idx] = np.exp(logr_tr[idx])  # visited -> priority
        elif arm == "teacher" and len(off_idx):
            # Teacher reward, eqs. (4)-(6) of the module docstring:
            # d_s is the student's TB discrepancy (post-update
            # student, detached) on the teacher's OWN samples.
            t_points = points[:len(off_idx)]
            with torch.no_grad():
                d_s = (logr[:len(off_idx)]
                       - student.log_z(feats1)[0]
                       - student.log_pf_points(
                           t_points, feats1.expand(len(off_idx), -1)))
            # eq. (5): weight UNDERSAMPLED states, where delta > 0
            # means the forward flow is too small. Without this the
            # teacher chases oversampled states just as hard as missed
            # ones, which is the opposite of what it is for.
            wgt = 1.0 + teacher_c * (d_s > 0).to(d_s.dtype)
            logr_t = torch.log(teacher_eps + wgt * d_s ** 2)
            if teacher_alpha:
                # eq. (6): mix in the student's own log-reward, so the
                # teacher aims at regions that are both high-loss and
                # high-reward.
                logr_t = logr_t + teacher_alpha * logr[:len(off_idx)]
            lp_t = teacher.log_pf_points(
                t_points, feats1.expand(len(off_idx), -1))
            dt = logr_t - teacher.log_z(feats1)[0] - lp_t
            t_loss = (dt ** 2).mean()
            t_opt.zero_grad()
            t_loss.backward()
            t_opt.step()
        elif arm == "contrastive":
            uniq = np.unique(idx)
            d_pos = np.unique(np.concatenate([d_pos, uniq[~dead[uniq]]]))
            if len(d_pos) > buf_cap:  # evict lowest reward (keep best)
                d_pos = d_pos[np.argsort(logr_tr[d_pos])[-buf_cap:]]
            d_neg = np.concatenate([d_neg, uniq[dead[uniq]]])[-buf_cap:]

        seen[idx] = True
        shell_hits += int(shell[idx].sum())
        samples += len(idx)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            lp = student.log_pf_grid(feats1[0]).cpu().numpy().reshape(-1)
            pol = np.exp(lp)
            found = int((seen & is_mode).sum())
            n_modes = int(is_mode.sum())
            row = {
                "step": step, "loss": float(loss.item()),
                "samples": samples, "modes_found": found,
                "n_modes": n_modes,
                # training discovery: what the run saw, injected
                # states included
                "frac_modes": found / max(n_modes, 1),
                "train_discovery": found / max(n_modes, 1),
                # final-policy coverage: properties of the trained
                # sampler alone, exact (the terminating distribution
                # is enumerable, so no sampling is needed to measure
                # it)
                "policy_mass_modes": float(pol[is_mode].sum()),
                "policy_coverage": expected_coverage(pol, is_mode,
                                                     samples),
                "n_modes_live": int(is_live.sum()),
                "policy_coverage_live": expected_coverage(pol, is_live,
                                                          samples),
                "target_coverage_live": expected_coverage(target, is_live,
                                                          samples),
                # the same budget applied to the exact target: the
                # ceiling any sampler can reach here, and the
                # reference policy_coverage must be read against
                "target_coverage": expected_coverage(target, is_mode,
                                                     samples),
                "reward_queries": samples,
                "edge_share": shell_hits / samples,
                "l1": l1(pol, target),
                "wall_s": time.time() - t0}
            if arm == "contrastive":
                row["loss_aux"] = loss_aux
            history.append(row)
    return student, history


def samples_to_frac(history: list[dict], frac: float) -> float:
    """First 'samples' value at which frac_modes >= frac (NaN if never
    reached; reported, not hidden)."""
    for h in history:
        if h["frac_modes"] >= frac:
            return float(h["samples"])
    return float("nan")
