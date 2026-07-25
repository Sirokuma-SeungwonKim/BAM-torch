"""
CEP Block v2 (BAM-torch CD Ver.2, 2026-05-21~) — CENT2 full paradigm

Energy structure (Ver.2):
    E_total = E_SR + U_CENT
    U_CENT = U_CEP_SELF + U_SLR
        U_CEP_SELF = sum_i [chi_i q_i + 0.5 J_i q_i^2]    (atom self-energy, Parr-Pearson)
        U_SLR      = 0.5 sum_{i != j} k_e q_i q_j kappa(r_ij)  (pairwise long-range, shielded)

CEP Lagrange (charge conservation hard constraint, unchanged from Ver.1):
    lambda = (Q_total + sum_i chi_i/J_i) / sum_i (1/J_i)
    q_i    = (lambda - chi_i) / J_i

NOTE: U_SLR adds pairwise long-range Coulomb (missing in Ver.1). chi/J 학습이 paper-level
정확도 도달했으므로 U_SLR 항이 absolute energy reference 를 추가 제공 →
scale_shift target-leak (bugs #20) 의 −0.42 eV bias 더 강하게 줄어들 것으로 예상.

slr_kernel options:
    'cent2' : CENT2 paper Eq.5 shielded Green's function
              kappa(r) = 1/r - exp(-(lambda r)^n) / r
              n=4, lambda DEFAULT = 0.340 Å^-1 (paper optimal 0.18 a_0^-1)
              paper examined λ ∈ {0.16, 0.18, 0.20} a_0^-1
                            ≡ {0.302, 0.340, 0.378} Å^-1
    'erf'   : Gaussian charge smearing (simpler, EEM/QEq lineage)
              kappa(r) = erf(r/sigma) / r,  sigma ≈ 1.0 Å
    'none'  : disable U_SLR (= Ver.1 behavior, ablation control)

UNITS:
    All distances [Å], charges [e], energies [eV].
    Paper uses atomic units (a_0, Hartree, e). Conversion: 1 a_0 = 0.529177 Å.
    Note that paper's `lambda^n r^n = (lambda·r)^n` (lambda and r are scalars).

Reference:
    Khajehpasha et al., Phys. Rev. B 105, 144106 (2022)  — CENT2 (Eq.2, Eq.5)
    Ghasemi & Goedecker, J. Chem. Phys. 154, 074107 (2021) — CENT1
    Mortier et al., J. Am. Chem. Soc. 108, 4315 (1986)   — EEM (erf kernel lineage)
    Parr & Pearson, J. Am. Chem. Soc. 105, 7512 (1983)   — chemical hardness J
"""

import math
import torch
import torch.nn as tnn
from typing import Dict, Optional

from e3nn import o3

from bam_torch.utils.scatter import scatter_sum


# Coulomb constant in eV·Å/e² (CODATA, matches PhysNet default)
K_E_eV_A = 14.399645351950548


class CEPBlockV2(tnn.Module):
    """
    CEP Block Ver.2 — CENT2 full (self + U_SLR).

    Args:
        irreps_in   : irreps of input node features (used to extract scalar 0e components)
        num_species : number of element types (size of J_i parameter table)
        hidden_dim  : hidden dim of chi_mlp (default 64)
        slr_kernel  : 'cent2' | 'erf' | 'none' (default 'cent2')
        slr_lambda  : λ for cent2 kernel [Å^-1] (default 0.340 = paper optimal 0.18 a_0^-1)
        slr_n       : n for cent2 kernel (default 4, CENT2 paper)
        slr_sigma   : σ for erf kernel [Å] (default 1.0)
        slr_cutoff  : optional long-range cutoff in Å (None = all-pair within same graph)
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        num_species: int,
        hidden_dim: int = 64,
        slr_kernel: str = 'cent2',
        slr_lambda: float = 0.340,        # Å^-1 (= 0.18 a_0^-1, paper optimal)
        slr_n: int = 4,
        slr_sigma: float = 1.0,
        slr_cutoff: Optional[float] = None,
    ):
        super().__init__()

        # ── scalar (l=0, parity-even) component dimension ────────────────
        self.scalar_dim: int = irreps_in.count(o3.Irrep(0, 1))

        # [FIX inherited from Ver.1 bugs #23] 0e block 정확 mask
        scalar_mask = torch.zeros(irreps_in.dim, dtype=torch.bool)
        for slice_obj, (_mul, ir) in zip(irreps_in.slices(), irreps_in):
            if ir == o3.Irrep(0, 1):  # parity-even scalar 만
                scalar_mask[slice_obj] = True
        self.register_buffer('_scalar_mask', scalar_mask, persistent=False)

        # ── chi prediction MLP (env-dep electronegativity, unchanged from Ver.1) ──
        self.chi_mlp = tnn.Sequential(
            tnn.Linear(self.scalar_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, 1),
        )

        # ── J (per-element hardness, softplus, unchanged from Ver.1) ──
        self.J_raw = tnn.Parameter(torch.ones(num_species))

        # ── SLR kernel config (Ver.2 new) ────────────────────────────────
        assert slr_kernel in ('cent2', 'erf', 'none'), \
            f"slr_kernel must be 'cent2'|'erf'|'none', got {slr_kernel}"
        self.slr_kernel = slr_kernel
        self.slr_lambda = slr_lambda
        self.slr_n = slr_n
        self.slr_sigma = slr_sigma
        self.slr_cutoff = slr_cutoff

    def _compute_kappa(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise kernel κ(r) for U_SLR. Accepts any shape (sparse [E] or dense).

        cent2 : CENT2 Eq.5  κ(r) = (1 - exp(-(λr)^n)) / r
        erf   : Gaussian    κ(r) = erf(r/σ) / r
        """
        if self.slr_kernel == 'cent2':
            damping = torch.exp(-(self.slr_lambda * r) ** self.slr_n)
            return (1.0 - damping) / r
        elif self.slr_kernel == 'erf':
            return torch.erf(r / self.slr_sigma) / r
        else:  # 'none'
            return torch.zeros_like(r)

    def _compute_u_slr(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Compute U_SLR = 0.5 sum_{i != j (same graph)} k_e q_i q_j κ(r_ij).

        Edge-list sparse implementation:
          1. Build (i, j) index pairs once under no_grad (no autograd memory)
             — same-graph, i ≠ j, optional cutoff filter.
          2. Compute r_ij, κ, q_i·q_j only over E ≈ batch × N × (N-1) pairs
             (vs N_total² dense matrix → ~8× memory reduction for batch=32).
          3. scatter_sum per graph.

        Memory: O(E) instead of O(N_total²). Each tensor in the autograd graph
        is [E] (≈ 10K floats for batch=32, N≈18) instead of [N_total, N_total]
        (~332K floats).

        Returns: [num_graphs] U_SLR per graph.
        """
        if self.slr_kernel == 'none':
            return torch.zeros(num_graphs, device=q.device, dtype=q.dtype)

        N = positions.shape[0]

        # ── 1) Build pair indices (no autograd memory) ────────────────────
        with torch.no_grad():
            same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)   # [N, N]
            not_self = ~torch.eye(N, dtype=torch.bool, device=batch.device)
            mask = same_graph & not_self
            idx = mask.nonzero(as_tuple=False)                      # [E, 2]
        i = idx[:, 0]
        j = idx[:, 1]

        # ── 2) Edge-wise distances (autograd path) ────────────────────────
        r_ij = (positions[i] - positions[j]).norm(dim=-1)           # [E]

        # Optional long-range cutoff (filter under no_grad, then index)
        if self.slr_cutoff is not None:
            with torch.no_grad():
                keep = r_ij <= self.slr_cutoff
            i = i[keep]; j = j[keep]
            r_ij = r_ij[keep]

        # ── 3) Pair contribution: 0.5 * k_e * q_i * q_j * κ(r_ij) ─────────
        # 0.5 factor accounts for (i,j) + (j,i) double-count (both included)
        kappa = self._compute_kappa(r_ij)                            # [E]
        u_slr_pair = 0.5 * K_E_eV_A * q[i] * q[j] * kappa            # [E]

        # ── 4) scatter_sum per graph (same-graph guaranteed via mask) ─────
        u_slr = scatter_sum(u_slr_pair, batch[i], dim=0, dim_size=num_graphs)  # [G]
        return u_slr

    def forward(
        self,
        node_feats: torch.Tensor,           # [N, irreps_dim]
        species: torch.Tensor,               # [N]
        total_charge: torch.Tensor,          # [G]
        batch: torch.Tensor,                 # [N]
        num_graphs: int,
        positions: Optional[torch.Tensor] = None,  # [N, 3] — Ver.2 new for U_SLR
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys:
            atomic_charges, chi, J, U_CEP_SELF, U_SLR, U_CENT, total_charge

        Notes:
            U_CENT = U_CEP_SELF + U_SLR  (진짜 CENT2 paper U_CENT 정의)
            positions 필수 if slr_kernel != 'none'.
        """
        # ── χ prediction (parity-even scalar 채널만, Ver.1 fix 유지)
        scalar_feats = node_feats[:, self._scalar_mask]
        chi = self.chi_mlp(scalar_feats).squeeze(-1)           # [N]

        # ── J = softplus(J_raw)
        J = tnn.functional.softplus(self.J_raw)[species]       # [N]

        # ── CEP Lagrange analytical solution (Ver.1 동일)
        eps = 1e-8
        inv_J = 1.0 / (J + eps)
        sum_chi_over_J = scatter_sum(chi * inv_J, batch, dim=0, dim_size=num_graphs)
        sum_inv_J = scatter_sum(inv_J, batch, dim=0, dim_size=num_graphs)
        lam = (total_charge + sum_chi_over_J) / (sum_inv_J + eps)
        lam_per_atom = lam[batch]

        # q_i = (λ - χ_i) / J_i
        q = (lam_per_atom - chi) * inv_J                       # [N]

        # ── U_CEP_SELF (atom self-energy, Ver.1 동일)
        U_CEP_SELF_per_atom = chi * q + 0.5 * J * q.pow(2)
        U_CEP_SELF = scatter_sum(U_CEP_SELF_per_atom, batch, dim=0, dim_size=num_graphs)

        # ── U_SLR (Ver.2 new, pairwise long-range)
        if self.slr_kernel != 'none':
            assert positions is not None, \
                "positions required when slr_kernel != 'none'"
            U_SLR = self._compute_u_slr(q, positions, batch, num_graphs)
        else:
            U_SLR = torch.zeros_like(U_CEP_SELF)

        # ── U_CENT = U_CEP_SELF + U_SLR  (진짜 CENT2 paper U_CENT 정의 부활)
        U_CENT = U_CEP_SELF + U_SLR

        # Conservation verification
        q_total = scatter_sum(q, batch, dim=0, dim_size=num_graphs)

        return {
            "atomic_charges": q,
            "chi": chi,
            "J": J,
            "U_CEP_SELF": U_CEP_SELF,
            "U_SLR": U_SLR,
            "U_CENT": U_CENT,          # Ver.2 에서 진짜 의미 (self + SLR)
            "total_charge": q_total,
        }
