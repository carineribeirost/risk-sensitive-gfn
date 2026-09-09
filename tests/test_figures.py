"""Figure/table builders: every F1–F5 function renders a PDF from
synthesized minimal inputs (no dependency on committed results/)."""

import csv
import importlib.util
import json
import pathlib

spec = importlib.util.spec_from_file_location(
    "make_figures",
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fake_o1(d):
    _csv(d / "o1_caseB_sep.csv",
         [{"world": w, "rho_p": rp, "rho_m": rm,
           "tv_to_shared": 0.02 + 0.1 * rp + 0.05 * rm}
          for w in range(2) for rp in (0.0, 0.5) for rm in (0.0, 0.5)])
    _csv(d / "o1_caseC_sep.csv",
         [{"world": w, "rho": rho, "delta": dl,
           "tv_to_shared": 0.01 + dl}
          for w in range(2) for rho in (0.0, 0.5)
          for dl in (0.0, 0.05, 0.1)])
    _csv(d / "o1_caseD_sep.csv",
         [{"world": w, "origin": o, "rho_hi": 1.0,
           "tv_to_shared": 0.06 + 0.01 * o,
           "tv_to_shared_full_split": 0.09}
          for w in range(2) for o in range(3)])


def _fake_o2(d, case, l1, fl):
    _csv(d / f"o2_case{case}.csv",
         [{"case": case, "world": w, "train_seed": s,
           "heldout_l1": l1 + 0.01 * s, "mc_floor_l1": fl}
          for w in range(3) for s in range(2)])


def test_all_figures_render(tmp_path, monkeypatch):
    """Every figure/table builder renders its output file from synthesized
    minimal CSV/JSON inputs."""
    monkeypatch.setattr(mf, "OUT", tmp_path / "figs")
    o1 = tmp_path / "o1"
    _fake_o1(o1)
    mf.fig_separability(str(o1))

    o2 = tmp_path / "o2"
    for c, v in (("A", 0.1), ("B", 0.5), ("C", 0.3), ("D", 0.15)):
        _fake_o2(o2, c, v, 0.2)
    tied = tmp_path / "tied"
    _fake_o2(tied, "B", 0.45, 0.2)
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    for c in "AB":
        (oracle / f"oracle_gap_case{c}_summary.json").write_text(
            json.dumps({"case": c, "tb": {"mean": 0.5},
                        "exact_kl": {"mean": 0.2},
                        "gap_tb_minus_kl": {"mean": 0.3},
                        "permutation": {"p_value": 0.02},
                        "mc_floor_mean": 0.2}))
    mf.fig_amortization([str(o2)], str(tied), str(oracle))

    asym = tmp_path / "asym.json"
    asym.write_text(json.dumps({"confirmatory_B": {
        "per_world": {str(w): {"abs_log_kappa_ratio": 0.1 * w,
                               "mean_delta_stress_p05": 0.01,
                               "mean_delta_contam_worst": 0.008}
                      for w in range(4)},
        "delta_stress_p05": {"mean": 0.01, "p_value": 0.03},
        "dose_response": {"spearman_rho": 0.5, "p_value": 0.2}}}))
    mf.fig_utility(str(asym))

    mf.fig_scale(str(o2), str(o2), str(o1), str(o1),
                 str(asym), str(asym))

    w33 = tmp_path / "w33_runs.csv"
    _csv(w33,
         [{"case": "B", "sparsity": s, "world": w, "arm": a, "seed": 0,
           "frac_modes": 0.5 + (0.2 if a == "teacher" else 0.0),
           "final_l1": 0.3}
          for s in (1.0, 4.0) for w in range(2)
          for a in ("onpolicy", "teacher")])
    mf.fig_reproduction(str(w33))

    rec = tmp_path / "recovery.csv"
    rec_rows = []
    for case in ("A", "B"):
        for w in range(2):
            for arm, model, lv, sig, mass in (
                    ("true", "none", 0.0, 0.0, 0.9),
                    ("crude", "none", 0.0, 0.0, 0.6),
                    ("naive", "crash", 0.05, 0.0, 0.4),
                    ("robust", "crash", 0.05, 0.1, 0.7),
                    ("winsor", "crash", 0.05, 0.0, 0.75),
                    ("naive", "crash", 0.2, 0.0, 0.2),
                    ("robust", "crash", 0.2, 0.1, 0.35),
                    ("winsor", "crash", 0.2, 0.0, 0.5)):
                rec_rows.append({"case": case, "world": w,
                                 "model": model, "level": lv,
                                 "beta": 0.3, "rho": 0.2, "arm": arm,
                                 "sigma": sig,
                                 "sat_mass_true": mass + 0.01 * w,
                                 "tv_true": 0.1, "n_skipped": 0})
    _csv(rec, rec_rows)
    mf.fig_recovery(str(rec))
    mf.fig_fallback(str(rec))

    mf.OUT.mkdir(parents=True, exist_ok=True)
    mf.tables(str(o1), [str(o2)], str(oracle))
    names = {p.name for p in (tmp_path / "figs").iterdir()}
    assert {"fig_separability.pdf", "fig_amortization.pdf",
            "fig_utility.pdf", "fig_scale.pdf",
            "fig_reproduction.pdf", "fig_recovery.pdf",
            "fig_fallback.pdf", "tab1_separability.tex",
            "tab2_amortization.tex"} <= names
