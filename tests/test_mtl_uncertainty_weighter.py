"""
Unit test for the multi-task UncertaintyWeighter (interference mitigation).

Checks:
  [1] forward reproduces the Kendall formula  L = sum_i [0.5 exp(-s_i) L_i + 0.5 s_i]
  [2] at log_var=0 it reduces to 0.5 * sum_i L_i (unit-weight baseline)
  [3] None-valued / missing task losses are ignored
  [4] gradients flow to the log-variances and an optimizer drives them toward the
      analytic optimum s_i* = log(L_i) for fixed losses (high-loss task -> down-weighted)

Run:  python tests/test_mtl_uncertainty_weighter.py   (also importable as pytest)
"""
import math
import torch

from bam_torch.charge_dependent.training.mtl import UncertaintyWeighter


def test_forward_matches_kendall_formula():
    w = UncertaintyWeighter(("e", "f", "q"))
    with torch.no_grad():
        w.log_var["e"].fill_(0.0)
        w.log_var["f"].fill_(1.0)
        w.log_var["q"].fill_(-0.5)
    losses = {"e": torch.tensor(2.0), "f": torch.tensor(3.0), "q": torch.tensor(0.5)}
    out = w(losses).item()
    expected = sum(
        0.5 * math.exp(-s) * L + 0.5 * s
        for s, L in [(0.0, 2.0), (1.0, 3.0), (-0.5, 0.5)]
    )
    assert abs(out - expected) < 1e-6, f"{out} != {expected}"


def test_unit_weight_at_zero_logvar():
    w = UncertaintyWeighter(("e", "f", "q"))   # init_log_var=0
    losses = {"e": torch.tensor(1.0), "f": torch.tensor(2.0), "q": torch.tensor(4.0)}
    out = w(losses).item()
    assert abs(out - 0.5 * (1.0 + 2.0 + 4.0)) < 1e-6, out


def test_missing_and_none_losses_ignored():
    w = UncertaintyWeighter(("e", "f", "q"))
    # f missing entirely, q is None -> only e contributes
    out = w({"e": torch.tensor(2.0), "q": None}).item()
    assert abs(out - 0.5 * 2.0) < 1e-6, out


def test_optimizer_drives_logvar_to_analytic_optimum():
    # For fixed losses, argmin_s [0.5 exp(-s) L + 0.5 s] => exp(-s) L = 1 => s* = log(L).
    w = UncertaintyWeighter(("e", "f", "q"))
    fixed = {"e": torch.tensor(1.0), "f": torch.tensor(1.0), "q": torch.tensor(100.0)}
    opt = torch.optim.Adam(w.parameters(), lr=0.05)
    for _ in range(4000):
        opt.zero_grad()
        loss = w(fixed)
        loss.backward()
        opt.step()
    s = w.log_vars()
    # high-loss task q must be down-weighted relative to e/f
    assert s["q"] > s["e"] + 2.0, s
    # converged near analytic optima s_e*=log1=0, s_q*=log100~=4.605
    assert abs(s["e"] - math.log(1.0)) < 0.15, s
    assert abs(s["q"] - math.log(100.0)) < 0.15, s
    # effective weights: q strongly down-weighted vs e
    wts = w.weights()
    assert wts["q"] < wts["e"], wts


def _run_all():
    test_forward_matches_kendall_formula()
    test_unit_weight_at_zero_logvar()
    test_missing_and_none_losses_ignored()
    test_optimizer_drives_logvar_to_analytic_optimum()
    w = UncertaintyWeighter(("e", "f", "q"))
    fixed = {"e": torch.tensor(1.0), "f": torch.tensor(1.0), "q": torch.tensor(100.0)}
    opt = torch.optim.Adam(w.parameters(), lr=0.05)
    for _ in range(4000):
        opt.zero_grad(); w(fixed).backward(); opt.step()
    print("[PASS] UncertaintyWeighter unit tests")
    print(f"       learned log-vars s = {w.log_vars()}  (analytic: e=0.0, q={math.log(100):.3f})")
    print(f"       effective weights  = {w.weights()}  (q down-weighted vs e)")


if __name__ == "__main__":
    _run_all()
