# Reproducing the results

Commands run from the repository root. Each folder under
`paper/results/` carries the CLI args used (`*_args.json`) plus the
per-case CSVs and summaries.

## Table 3 — learnability, calibration family

```bash
python scripts/launch_o2.py --cases A B C D --worlds 4 --world-seed0 0 \
    --seeds 3 --H 32 --steps 12000 --cond-pool 256 --logit-floor -25.0 \
    --jobs 10 --threads-per-job 2 --out results/o2-dev
```

Oracle column, per case:

```bash
python scripts/run_oracle_gap.py --case A --worlds 4 --world-seed0 0 \
    --seeds 3 --H 32 --steps 12000 --cond-pool 256 --logit-floor -25.0 \
    --device cuda --out results/oracle-gap
```

`paper/results/table3_and_oracle/`

## Table E.1 — resolution and depth

Size axis, case C at H=16 and H=64:

```bash
python scripts/run_o2.py --case C --worlds 4 --world-seed0 0 --seeds 3 \
    --H 16 --steps 12000 --cond-pool 256 --logit-floor -25.0 \
    --device cuda --out results/o2-H16
```

H=64: same, `--H 64 --out results/o2-H64`.

Depth axis, H=8/d=4, all cases:

```bash
python scripts/launch_o2.py --cases A B C D --worlds 4 --world-seed0 0 \
    --seeds 3 --H 8 --d 4 --steps 12000 --cond-pool 256 --logit-floor -25.0 \
    --jobs 16 --threads-per-job 2 --out results/o2-d4
```

`paper/results/table_e1_scaling/`

## Figures 2, 3, D.1–D.8, Table 4 — exploration under sparsity

```bash
python scripts/launch_w33.py --out results/w33
```

Cases, arms, sparsity (1.0, 4.0), and the deployment-scale sequence
family (H=4, d=8, geometry=sequence) are the script's defaults. Mode
coverage, dead-share, and per-arm L1 collect into
`paper/results/figures_2_3_D1-D8_table4/data.json`.

## Amortization ablations (§3.3, §4.2)

Per-condition experts against the conditional student:

```bash
python scripts/run_experts.py --case A --worlds 4 --world-seed0 0 \
    --seeds 3 --H 32 --steps 4500 --cond-pool 256 --logit-floor -25.0 \
    --device cuda --out results/experts
```

On/off-policy mix sweep:

```bash
python scripts/run_mix.py --case A --worlds 4 --world-seed0 0 --seeds 3 \
    --H 32 --steps 12000 --cond-pool 256 --logit-floor -25.0 \
    --device cuda --out results/mix
```

Both repeat per case, `--case B/C/D`.
`paper/results/experts_and_mix_ablations/`
