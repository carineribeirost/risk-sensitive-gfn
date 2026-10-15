"""Conditional forward policy P_F(· | c) and log Z(c),
d-generalized for arbitrary trajectory length.

Architecture: condition encoder MLP → FiLM-conditioned MLP trunk → one
H-way head shared across all d steps. The state before step t is the
chosen prefix (x₀..x_{t−1}); the trunk consumes the SUM of
position-tagged coordinate embeddings: one `coord_embed` table with a
row per (position s < d−1, value v) at index s·H + v, plus a start
token (index (d−1)·H) for the empty prefix. Because construction order
is canonical and fixed, the tagged sum determines the next position
uniquely, so no separate step embedding is needed, and at d = 2 the
table is exactly the former `x1_embed` (H value rows + null), making
the generalization bit-for-bit identical to the 2-step implementation
(verified by tests/test_w31_regression.py). log Z is a
separate head on the condition embedding: it must be a function of c,
not a scalar, for conditional trajectory balance to be well-posed.

With canonical construction order and P_B = 1 there is exactly one
trajectory per point, so the terminating density is the product of d
softmaxes and is enumerable exactly by stepwise expansion of a (H^t,)
log-prob tensor (log_pf_grid); evaluation needs no Monte Carlo on the
policy side. Keep d·H^d ≲ 10⁶.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """One residual trunk block, feature-wise linearly modulated by the
    condition embedding (FiLM: scale/shift applied after the linear +
    layer-norm, before the SiLU nonlinearity and residual add)."""

    def __init__(self, dim: int, cond_dim: int):
        """Args:
            dim: Trunk hidden width.
            cond_dim: Width of the condition embedding fed to FiLM.
        """
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            h: Trunk hidden state, shape (..., dim).
            e: Condition embedding, shape (..., cond_dim).

        Returns:
            Updated hidden state, same shape as `h`.
        """
        scale, shift = self.film(e).chunk(2, dim=-1)
        u = self.norm(self.lin(h)) * (1.0 + scale) + shift
        return h + F.silu(u)


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(),
                         nn.Linear(d_hidden, d_out))


class ConditionalPolicy(nn.Module):
    """Conditional forward policy P_F(· | c) plus the log Z(c) head.

    See the module docstring for the construction-order and embedding
    scheme; `sample`/`log_pf_points`/`log_pf_grid` are the three ways
    to use a trained instance (draw, score given points, or enumerate
    the exact density for one condition).
    """

    def __init__(self, H: int, n_cond_features: int = 4,
                 cond_dim: int = 128, dim: int = 256, depth: int = 4,
                 x1_dim: int = 64, d: int = 2, with_flows: bool = False):
        """Args:
            H: Grid side length (values per coordinate).
            n_cond_features: Width of the input condition feature vector.
            cond_dim: Width of the encoded condition embedding.
            dim: Trunk hidden width.
            depth: Number of FiLM blocks in the trunk.
            x1_dim: Width of the coordinate embedding table (name kept
                from the 2-step implementation for config compatibility).
            d: Coordinates per point (trajectory length).
            with_flows: If True, also allocate the SubTB state-flow head.
        """
        # x1_dim: width of the coordinate embedding (name kept from the
        # 2-step implementation for config compatibility)
        super().__init__()
        self.H = H
        self.d = d
        self.null_id = (d - 1) * H  # start-token row (== H at d = 2)
        self.cond_enc = _mlp(n_cond_features, 128, cond_dim)
        self.coord_embed = nn.Embedding((d - 1) * H + 1, x1_dim)
        self.inp = nn.Linear(x1_dim, dim)
        self.blocks = nn.ModuleList(FiLMBlock(dim, cond_dim)
                                    for _ in range(depth))
        self.head = nn.Linear(dim, H)
        self.logz_head = _mlp(cond_dim, 128, 1)
        # SubTB state-flow head (opt-in; allocated LAST so default
        # policies stay parameter-identical to a policy without it;
        # the d=2 bit-for-bit golden suite gates this)
        if with_flows:
            self.flow_head = nn.Linear(dim, 1)

    def embed_cond(self, cond_feats: torch.Tensor) -> torch.Tensor:
        """Encode raw condition features into the embedding used by
        the trunk and the log Z head.

        Args:
            cond_feats: Condition features, shape (..., n_cond_features).

        Returns:
            Condition embedding, shape (..., cond_dim).
        """
        return self.cond_enc(cond_feats)

    def log_z(self, cond_feats: torch.Tensor) -> torch.Tensor:
        """log Z(c), the per-condition partition-function head.

        Args:
            cond_feats: Condition features, shape (B, n_cond_features).

        Returns:
            log Z values, shape (B,).
        """
        return self.logz_head(self.embed_cond(cond_feats)).squeeze(-1)

    def _trunk(self, state_emb: torch.Tensor,
               cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.inp(state_emb)
        for blk in self.blocks:
            h = blk(h, cond_emb)
        return h

    def _logits(self, state_emb: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        """Logits for the next coordinate given summed prefix embeddings."""
        return self.head(self._trunk(state_emb, cond_emb))

    def _prefix_emb(self, prefix: torch.Tensor) -> torch.Tensor:
        """Summed position-tagged embedding of a chosen prefix (B, t);
        t = 0 → the start token."""
        b, t = prefix.shape
        if t == 0:
            ids = torch.full((b,), self.null_id, dtype=torch.long,
                             device=prefix.device)
            return self.coord_embed(ids)
        offs = torch.arange(t, device=prefix.device) * self.H
        return self.coord_embed(prefix + offs).sum(dim=1)

    def log_pf_points(self, points: torch.Tensor,
                      cond_feats: torch.Tensor) -> torch.Tensor:
        """log P_F(x | c) for paired (points (B,d) long, cond_feats (B,F))."""
        e = self.embed_cond(cond_feats)
        rows = torch.arange(points.shape[0], device=points.device)
        total = None
        for t in range(self.d):
            lp = F.log_softmax(
                self._logits(self._prefix_emb(points[:, :t]), e), dim=-1)
            term = lp[rows, points[:, t]]
            total = term if total is None else total + term
        return total

    def log_pf_points_flows(self, points: torch.Tensor,
                            cond_feats: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step log P_F terms (B, d) and intermediate state log-flows
        (B, d−1) for SubTB (requires with_flows=True). The flow of the
        length-t prefix (0 < t < d) is read off the same trunk state
        that produces step t's logits."""
        e = self.embed_cond(cond_feats)
        rows = torch.arange(points.shape[0], device=points.device)
        lps, flows = [], []
        for t in range(self.d):
            h = self._trunk(self._prefix_emb(points[:, :t]), e)
            lp = F.log_softmax(self.head(h), dim=-1)
            lps.append(lp[rows, points[:, t]])
            if t > 0:
                flows.append(self.flow_head(h).squeeze(-1))
        return (torch.stack(lps, dim=1),
                torch.stack(flows, dim=1) if flows else
                torch.zeros(points.shape[0], 0, device=points.device))

    @torch.no_grad()
    def log_pf_grid(self, cond_feats_one: torch.Tensor) -> torch.Tensor:
        """Exact terminating log-density over all of X for ONE condition,
        shape (H,)*d; flatten in canonical order to compare with p*."""
        dev = cond_feats_one.device
        H, d = self.H, self.d
        e1 = self.embed_cond(cond_feats_one.unsqueeze(0))
        start = self.coord_embed(torch.tensor([self.null_id], device=dev))
        acc = F.log_softmax(self._logits(start, e1), dim=-1)[0]  # (H,)
        vals = torch.arange(H, device=dev)
        emb = self.coord_embed(vals)  # prefixes of length 1, (H, E)
        for t in range(1, d):
            lp = F.log_softmax(
                self._logits(emb, e1.expand(emb.shape[0], -1)), dim=-1)
            acc = (acc.reshape(-1)[:, None] + lp).reshape(-1)  # (H^{t+1},)
            if t < d - 1:
                nxt = self.coord_embed(t * H + vals)
                emb = (emb[:, None, :] + nxt[None, :, :]
                       ).reshape(-1, emb.shape[-1])
        return acc.reshape((H,) * d)

    def log_pf_grid_train(self, cond_feats: torch.Tensor) -> torch.Tensor:
        """Batched exact terminating log-density, shape (B,)+(H,)*d, WITH
        gradients: the same softmax composition as log_pf_grid, for
        losses that need the full grid density (exact-KL).
        Σ_t H^t trunk rows per condition."""
        dev = cond_feats.device
        b, H, d = cond_feats.shape[0], self.H, self.d
        e = self.embed_cond(cond_feats)
        null = torch.full((b,), self.null_id, dtype=torch.long, device=dev)
        acc = F.log_softmax(
            self._logits(self.coord_embed(null), e), dim=-1)  # (B, H)
        vals = torch.arange(H, device=dev)
        emb = self.coord_embed(vals)  # (H, E), shared across conditions
        for t in range(1, d):
            m = emb.shape[0]  # H^t prefixes
            x = emb.unsqueeze(0).expand(b, m, -1).reshape(b * m, -1)
            lp = F.log_softmax(
                self._logits(x, e.repeat_interleave(m, dim=0)),
                dim=-1).view(b, m, H)
            acc = (acc.view(b, m)[:, :, None] + lp).reshape(b, m * H)
            if t < d - 1:
                nxt = self.coord_embed(t * H + vals)
                emb = (emb[:, None, :] + nxt[None, :, :]
                       ).reshape(-1, emb.shape[-1])
        return acc.reshape((b,) + (H,) * d)

    @torch.no_grad()
    def sample(self, cond_feats: torch.Tensor,
               gen: torch.Generator) -> torch.Tensor:
        """On-policy points, one per row of cond_feats, shape (B, d).
        `gen` must live on the same device as the module (see
        train.train_policy)."""
        dev = cond_feats.device
        e = self.embed_cond(cond_feats)
        b = cond_feats.shape[0]
        null = torch.full((b,), self.null_id, dtype=torch.long, device=dev)
        state = self.coord_embed(null)
        xs = []
        for t in range(self.d):
            p = F.softmax(self._logits(state, e), dim=-1)
            x = torch.multinomial(p, 1, generator=gen).squeeze(-1)
            xs.append(x)
            if t < self.d - 1:
                step = self.coord_embed(t * self.H + x)
                state = step if t == 0 else state + step
        return torch.stack(xs, dim=-1)
