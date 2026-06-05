"""
Multi-task loss weighting for charge-dependent training (interference mitigation).

Motivation (I-37, 2026-05-29):
    In the CENT2-full model the atomic charge q feeds BOTH the charge loss L_Q (NPA
    target) AND the total energy E_total = E_SR + U_CENT (and hence the force loss L_F
    via autograd, since F = -dE_total/dx). With FIXED loss weights (enr/frc/chg_lambda)
    the SHARED parameters (chi_mlp, hardness J, and the backbone) receive conflicting
    gradients from L_Q vs L_E/L_F — a multi-task trade-off that shows up as the charge
    MAE dose-response (none < cent2 < erf).

    This module provides Kendall et al. (2018) homoscedastic-uncertainty weighting as an
    OPT-IN alternative to the fixed lambdas. Each task gets a learnable log-variance
    s_i = log(sigma_i^2) and the combined loss is

        L = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]

    so the optimizer LEARNS the relative task weights instead of using hand-tuned lambdas.
    The +0.5*s_i term regularizes s_i (prevents the trivial s_i -> +inf collapse).
    The effective weight on task i is exp(-s_i); s_i = 0 reproduces unit weights.

    This complements the kernel-parameter route (slr_n, "option B"): option B reduces the
    NN-scale source of the conflict, uncertainty weighting balances whatever conflict
    remains at the optimization level. It does NOT remove the coupling (that would require
    use_cent_energy=False, which discards the charge->energy physics).

Reference:
    Kendall, Gal, Cipolla. "Multi-Task Learning Using Uncertainty to Weigh Losses for
    Scene Geometry and Semantics." CVPR 2018.
"""
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class UncertaintyWeighter(nn.Module):
    """Learnable homoscedastic-uncertainty weighting over named task losses.

    Args:
        task_keys:    iterable of task names, e.g. ("e", "f", "q"). One learnable
                      log-variance parameter is created per task.
        init_log_var: initial value of every log-variance (0.0 => unit weight, i.e.
                      identical to the fixed-weight loss at step 0 when all lambdas=1).
    """

    def __init__(self, task_keys: Iterable[str], init_log_var: float = 0.0):
        super().__init__()
        self.task_keys = list(task_keys)
        if not self.task_keys:
            raise ValueError("UncertaintyWeighter requires at least one task key")
        self.log_var = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(float(init_log_var)))
             for k in self.task_keys}
        )

    def forward(self, losses: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """Combine per-task scalar losses via Kendall uncertainty weighting.

        Only keys present in BOTH ``losses`` (with a non-None value) and ``self.log_var``
        are combined; any other entries are ignored. Returns a scalar tensor.
        """
        total: Optional[torch.Tensor] = None
        for k in self.task_keys:
            li = losses.get(k)
            if li is None:
                continue
            s = self.log_var[k]
            term = 0.5 * torch.exp(-s) * li + 0.5 * s
            total = term if total is None else total + term
        if total is None:
            raise ValueError(
                f"UncertaintyWeighter: no matching task losses to weight "
                f"(task_keys={self.task_keys}, got keys={list(losses)})"
            )
        return total

    @torch.no_grad()
    def weights(self) -> Dict[str, float]:
        """Current effective weight exp(-s_i) per task (for logging/inspection)."""
        return {k: float(torch.exp(-self.log_var[k])) for k in self.task_keys}

    @torch.no_grad()
    def log_vars(self) -> Dict[str, float]:
        """Current log-variance s_i per task (for logging/inspection)."""
        return {k: float(self.log_var[k]) for k in self.task_keys}
