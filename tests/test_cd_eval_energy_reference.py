"""
Regression test for the CD-evaluator energy-reference bug (fixed 2026-05-29).

BUG: the CD evaluators (cd_evaluator.py, cd_evaluator_v3.py; v4 inherits v3) reconstructed
the PREDICTION to full-DFT (preds['energy'] + node_enr_avg + e_corr) but stored
exact_energy = data['energy'], which is BASELINE-REMOVED for the CD dataloader
(cd_utils.py:223 subtracts sum_i enr_avg[Z_i]). So saved test_values had pred and exact on
DIFFERENT references -> a bogus ~-10953 eV bias, making the saved CD energy metric/scatter
invalid and CD-vs-base incomparable (the base evaluator stores full-DFT exact, so it was fine).

FIX: store exact_energy = data['energy'] + node_enr_avg (full-DFT), matching the prediction
reference and the base evaluator.

This test trains a tiny CD model on dummy data (CPU, 1 epoch), evaluates it, and asserts the
stored exact_energy and energy sit on the SAME reference (|mean(pred) - mean(exact)| small,
NOT ~1e4 eV). Pre-fix this assertion FAILS (bias ~1e4 eV); post-fix it passes.

Run:  python tests/test_cd_eval_energy_reference.py   (also importable as pytest)
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cd_training import generate_dummy_xyz, create_minimal_config
from bam_torch.charge_dependent.training.cd_trainer import CDTrainer
from bam_torch.charge_dependent.predicting import EVALUATOR_REGISTRY


def _scalar_means(seq):
    return np.array([float(np.asarray(t).mean()) for t in seq])


def test_cd_eval_energy_reference():
    wd = tempfile.mkdtemp(prefix="cd_evalref_")
    cwd0 = os.getcwd()
    try:
        os.chdir(wd)  # keep all evaluator/trainer outputs (test_values.pkl etc.) local
        xyz = os.path.join(wd, "dummy.xyz")
        generate_dummy_xyz(xyz, n_structures=30)
        cfg, _ = create_minimal_config(xyz, wd)
        cfg["device"] = "cpu"
        cfg["gpu-parallel"] = False
        cfg["NN"]["nepoch"] = 1
        cfg["NN"]["ema"] = False
        model_pkl = os.path.join(wd, "m.pkl")
        cfg["NN"]["fname_pkl"] = model_pkl

        # ── train a tiny CD model (saves model_pkl) ──
        CDTrainer(cfg, 0, 1).train()

        # ── evaluate on the same dummy set ──
        cfg["predict"] = {
            "evaluate_tag": False,
            "loss_config": {"energy_loss": "rmse", "force_loss": "rmse"},
            "ndata": 30,
            "fname_traj": xyz,
            "model": model_pkl,
            "fname_plog": os.path.join(wd, "predict.out"),
        }
        ev_cls = EVALUATOR_REGISTRY.get(cfg.get("trainer", "cd"))
        assert ev_cls is not None, f"no evaluator for trainer={cfg.get('trainer')}"
        ev_cls(cfg, 0, 1).evaluate()

        # ── locate the saved test_values.pkl ──
        tvp = None
        for root, _, files in os.walk(wd):
            for f in files:
                if f.endswith("test_values.pkl"):
                    tvp = os.path.join(root, f)
        assert tvp, f"test_values.pkl not produced under {wd}"

        tv = torch.load(tvp, map_location="cpu", weights_only=False)
        assert "energy" in tv and "exact_energy" in tv, f"keys={list(tv)}"
        pe = _scalar_means(tv["energy"])
        ee = _scalar_means(tv["exact_energy"])
        bias = abs(float(pe.mean() - ee.mean()))

        # the load-bearing assertion: same reference (a few eV), NOT ~1e4 eV
        assert bias < 50.0, (
            f"CD evaluator energy-reference MISMATCH: "
            f"|mean(pred) - mean(exact)| = {bias:.1f} eV (expected < 50; pre-fix ~1e4). "
            f"exact_energy must be reconstructed to full-DFT (+ node_enr_avg) to match pred."
        )
        print(f"[PASS] CD evaluator energy reference consistent: "
              f"|mean(pred) - mean(exact)| = {bias:.3f} eV  (mean pred={pe.mean():.2f}, exact={ee.mean():.2f})")
        return True
    finally:
        os.chdir(cwd0)


if __name__ == "__main__":
    test_cd_eval_energy_reference()
