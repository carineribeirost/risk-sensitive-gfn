"""Capture the d=2 golden numbers for the 2-step-to-d-step regression
gate.

Run ONCE against the old 2-step-hardcoded policy to freeze its exact
outputs; `test_w31_regression.py` then asserts the generalized
d-coordinate implementation reproduces every array exactly at d=2.
Committed alongside the golden .npz so the capture is auditable.

Usage: .venv/bin/python tests/make_w31_golden.py
"""

import numpy as np
import torch

from epgfn.policy import ConditionalPolicy
from epgfn.train import TrainConfig, train_policy
from epgfn.worlds import WorldConfig, sample_world

OUT = "tests/data/w31_golden.npz"


def main():
    """Freeze golden d=2 arrays (policy state, densities, samples, training
    trace) for the d-generalized-policy regression gate."""
    out = {}

    # 1. init RNG stream: full state_dict of a policy built under a seed
    torch.manual_seed(1234)
    pol = ConditionalPolicy(H=8, n_cond_features=4, cond_dim=32, dim=64,
                            depth=2, x1_dim=16)
    for k, v in pol.state_dict().items():
        out[f"sd::{k}"] = v.numpy().copy()

    # 2. exact grid density + point densities on fixed conditions
    torch.manual_seed(99)
    feats = torch.randn(3, 4)
    out["grid"] = np.stack([pol.log_pf_grid(feats[i]).numpy()
                            for i in range(3)])
    pts = torch.tensor([[0, 0], [3, 5], [7, 2], [5, 7]])
    with torch.no_grad():
        out["pts"] = pol.log_pf_points(
            pts, feats[0].unsqueeze(0).expand(4, -1)).numpy()
        out["grid_train"] = pol.log_pf_grid_train(feats).numpy()

    # 3. sampler RNG consumption
    gen = torch.Generator().manual_seed(7)
    out["samples"] = pol.sample(feats[1].unsqueeze(0).expand(32, -1),
                                gen).numpy()

    # 4. end-to-end training trace (world gen + pooled TB, tiny budget)
    world, _ = sample_world("A", WorldConfig(H=8), seed=0)
    out["world_g"] = world.g.copy()
    cfg = TrainConfig(steps=30, n_conds=2, n_points=16, eval_every=10,
                      seed=0, cond_pool=32,
                      net=dict(dim=64, depth=2, cond_dim=32))
    _, _, hist = train_policy(world, cfg)
    out["hist_loss"] = np.array([h["loss"] for h in hist])
    out["hist_l1"] = np.array([h["heldout_l1"] for h in hist])

    np.savez(OUT, **out)
    print(f"wrote {OUT}: {len(out)} arrays; "
          f"losses {out['hist_loss']}, l1 {out['hist_l1']}")


if __name__ == "__main__":
    main()
