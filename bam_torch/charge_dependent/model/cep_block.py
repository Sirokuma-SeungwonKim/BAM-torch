"""
CENT2-based Charge Equilibration Process (CEP) Block.

Phase 2 implementation:
  chi_i = MLP(scalar node features)  — ANN-predicted environment-dependent electronegativity
  J_i = softplus(J_raw[species])     — per-element chemical hardness (learnable parameter)

  CEP analytical solution (Lagrange multiplier method):
    min sum_i [ chi_i q_i + 0.5 J_i q_i^2 ]   s.t.  sum_i q_i = Q_total

    -> lambda = (Q_total + sum_i chi_i/J_i) / sum_i (1/J_i)   [per graph]
    -> q_i = (lambda - chi_i) / J_i                            [per atom]

  Charge conservation is mathematically guaranteed (hard constraint).

  U_CEP_SELF = sum_i [ chi_i q_i + 0.5 J_i q_i^2 ]
    (atom self-energy only; Parr-Pearson quadratic. Ver.1 simplification of CENT2.)

NOTE on naming (2026-05-21, Ver.1 rename):
  The output dict key was historically called "U_CENT" but is a misnomer:
  - CENT2 paper (Khajehpasha 2022) Eq.(2): U_CENT = self-energy + U_SLR (long-range Coulomb)
  - BAM-torch Ver.1: only the self-energy part (U_SLR not implemented)
  Therefore renamed to `U_CEP_SELF` (CEP atom Self-Energy).
  An alias `U_CENT` is preserved in the output dict for backward compatibility
  (deprecated; will be redefined in Ver.2 as `U_CENT = U_CEP_SELF + U_SLR`).

Reference:
  Khajehpasha et al., Phys. Rev. B 105, 144106 (2022)  — CENT2 (paradigm origin)
  Ghasemi & Goedecker, J. Chem. Phys. 154, 074107 (2021) — CENT1
  Rappé & Goddard, J. Phys. Chem. 95, 3358 (1991)      — QEq (Lagrange theory)
  Parr & Pearson, J. Am. Chem. Soc. 105, 7512 (1983)   — chemical hardness J
"""

import torch
import torch.nn as tnn
from typing import Dict

from e3nn import o3

from bam_torch.utils.scatter import scatter_sum


class CEPBlock(tnn.Module):
    """
    CENT2-based Charge Equilibration Process Block.

    Predicts environment-dependent electronegativity chi_i from node features,
    then analytically determines atomic charges q_i via Lagrange multiplier method.
    Charge conservation (sum q_i = Q_total) is always mathematically guaranteed.

    Args:
        irreps_in   : irreps of input node features (used to extract scalar components)
        num_species : number of element types (size of J_i parameter table)
        hidden_dim  : hidden dimension of chi_i prediction MLP (default 64)
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        num_species: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # scalar (l=0, even parity) component dimension
        self.scalar_dim: int = irreps_in.count(o3.Irrep(0, 1))

        # [FIX 2026-05-19, bugs #23] 0e block 의 정확한 mask 를 미리 계산.
        # 이전 코드 (node_feats[:, :scalar_dim]) 는 hidden_irreps 가
        # "32x0o+32x0e+..." 처럼 0o 먼저면 0o pseudoscalar (자연 0) 를 선택해버려
        # chi_mlp.0.weight 가 학습 안 되는 silent failure mode 가 있었음.
        scalar_mask = torch.zeros(irreps_in.dim, dtype=torch.bool)
        for slice_obj, (_mul, ir) in zip(irreps_in.slices(), irreps_in):
            if ir == o3.Irrep(0, 1):  # parity-even scalar 만 선택
                scalar_mask[slice_obj] = True
        self.register_buffer('_scalar_mask', scalar_mask, persistent=False)

        # chi_i prediction MLP (environment-dependent electronegativity)
        self.chi_mlp = tnn.Sequential(
            tnn.Linear(self.scalar_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, hidden_dim),
            tnn.SiLU(),
            tnn.Linear(hidden_dim, 1),
        )

        # J_i : per-element chemical hardness (learnable, softplus ensures positive)
        self.J_raw = tnn.Parameter(torch.ones(num_species))

    def forward(
        self,
        node_feats: torch.Tensor,       # [num_nodes, irreps_dim]
        species: torch.Tensor,           # [num_nodes]  element index
        total_charge: torch.Tensor,      # [num_graphs] total charge (Q_total)
        batch: torch.Tensor,             # [num_nodes]  batch index
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            atomic_charges : [num_nodes]  CEP-determined atomic charges q_i
            chi            : [num_nodes]  predicted atomic electronegativity chi_i
            J              : [num_nodes]  per-element chemical hardness J_i
            U_CEP_SELF     : [num_graphs] CEP atom self-energy (also exposed as `U_CENT` alias for backward compat)
            total_charge   : [num_graphs] sum q_i (for conservation verification, ~ input Q_total)
        """
        # ── chi_i prediction ───────────────────────────────────────────────
        # [FIX 2026-05-19, bugs #23] 0e block 정확한 mask 로 선택 (0o 채널 회피)
        scalar_feats = node_feats[:, self._scalar_mask]
        # [DEBUG] chi_mlp input 분포 (env _CEP_DEBUG=1 trigger)
        import os as _os
        if _os.environ.get('_CEP_DEBUG') and not _os.environ.get('_CEP_FEATS_PRINTED'):
            sf = scalar_feats.detach()
            nf = node_feats.detach()
            print(f"[DEBUG scalar_feats] shape={tuple(sf.shape)} L2={sf.norm().item():.4f}  (self.scalar_dim={self.scalar_dim})", flush=True)
            print(f"  per-elem stats: min={sf.min().item():.6f} max={sf.max().item():.6f} mean={sf.mean().item():+.6f} std={sf.std().item():.6f}", flush=True)
            print(f"  abs: mean={sf.abs().mean().item():.6f} median={sf.abs().median().item():.6f} max={sf.abs().max().item():.6f}", flush=True)
            print(f"  first 3 atoms × first 5 dims: {sf[:3, :5].tolist()}", flush=True)
            # node_feats 전체 channel-block 분석 (slicing 이 맞는지)
            print(f"[DEBUG node_feats total shape={tuple(nf.shape)}]", flush=True)
            n_dim = nf.shape[1]
            print(f"  block-wise L2 (32 dim 단위로 잘라봄, hidden_irreps 추정 32x0o+32x0e+32x1o+32x1e+32x2o+32x2e):", flush=True)
            for i in range(0, n_dim, 32):
                end = min(i + 32, n_dim)
                block = nf[:, i:end]
                print(f"    [{i:3d}:{end:3d}] L2={block.norm().item():9.4f}  mean={block.mean().item():+.6f}  abs_max={block.abs().max().item():.6f}", flush=True)
            # chi_mlp.0(x) output 분포 직접 측정
            with torch.no_grad():
                out0 = self.chi_mlp[0](sf)
                print(f"[DEBUG chi_mlp.0 output] L2={out0.norm().item():.4f} min={out0.min().item():.6f} max={out0.max().item():.6f} mean={out0.mean().item():+.6f}", flush=True)
                # SiLU(out0)
                out0_silu = torch.nn.functional.silu(out0)
                print(f"[DEBUG SiLU(chi_mlp.0)] L2={out0_silu.norm().item():.4f} min={out0_silu.min().item():.6f} max={out0_silu.max().item():.6f}", flush=True)
                # bias 만 vs weight 영향 비교
                bias_only = self.chi_mlp[0].bias.unsqueeze(0).expand_as(out0)
                weight_part = out0 - bias_only
                print(f"[DEBUG W·x part (out0 - bias)] L2={weight_part.norm().item():.4e}  ← weight 의 input 의존성 기여", flush=True)
                print(f"[DEBUG bias part]                L2={bias_only.norm().item():.4f}  ← bias 의 (atom-invariant) 기여", flush=True)
            _os.environ['_CEP_FEATS_PRINTED'] = '1'
        chi = self.chi_mlp(scalar_feats).squeeze(-1)          # [N]

        # ── J_i (softplus: always positive) ────────────────────────────────
        J = tnn.functional.softplus(self.J_raw)[species]      # [N]

        # ── CEP analytical solution (Lagrange multiplier) ──────────────────
        eps: float = 1e-8
        inv_J = 1.0 / (J + eps)                               # [N]

        # Per-graph aggregation
        sum_chi_over_J = scatter_sum(
            chi * inv_J, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]
        sum_inv_J = scatter_sum(
            inv_J, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]

        # lambda = (Q_total + sum chi/J) / sum (1/J)
        lam = (total_charge + sum_chi_over_J) / (sum_inv_J + eps)  # [G]
        lam_per_atom = lam[batch]                              # [N]

        # q_i = (lambda - chi_i) / J_i  ->  hard charge conservation
        q = (lam_per_atom - chi) * inv_J                      # [N]

        # ── U_CEP_SELF = sum_i [ chi_i q_i + 0.5 J_i q_i^2 ] ─────────────
        # NOTE (Ver.1): atom self-energy only. CENT2 paper U_CENT also contains
        # pairwise long-range U_SLR (shielded Green's function), not implemented here.
        # Ver.2 will add U_SLR and redefine `U_CENT = U_CEP_SELF + U_SLR`.
        U_CEP_SELF_per_atom = chi * q + 0.5 * J * q.pow(2)
        U_CEP_SELF = scatter_sum(
            U_CEP_SELF_per_atom, batch, dim=0, dim_size=num_graphs,
        )                                                      # [G]

        # For conservation verification
        q_total = scatter_sum(q, batch, dim=0, dim_size=num_graphs)

        import os as _os
        if _os.environ.get('_CEP_DEBUG') and not _os.environ.get('_CEP_INTERNAL_PRINTED'):
            print(f"[DEBUG CEP INTERNAL]", flush=True)
            print(f"  Q_input          = {total_charge[:3].tolist()}", flush=True)
            print(f"  sum_chi_over_J   = {sum_chi_over_J[:3].tolist()}", flush=True)
            print(f"  sum_inv_J        = {sum_inv_J[:3].tolist()}", flush=True)
            print(f"  lam              = {lam[:3].tolist()}", flush=True)
            print(f"  Σ q (predicted)  = {q_total[:3].tolist()}", flush=True)
            print(f"  chi range        = [{chi.min().item():.4f}, {chi.max().item():.4f}]", flush=True)
            print(f"  chi (per species) per first 3 atoms = {chi[:3].tolist()}", flush=True)
            print(f"  J (per atom) first 3 = {J[:3].tolist()}", flush=True)
            print(f"  inv_J first 3 = {inv_J[:3].tolist()}", flush=True)
            print(f"  q first 3 = {q[:3].tolist()}", flush=True)
            # Math check: sum_inv_J/(sum_inv_J + eps) factor
            factor = sum_inv_J[:3] / (sum_inv_J[:3] + eps)
            print(f"  conservation factor [sum_inv_J/(sum_inv_J+eps)] = {factor.tolist()}", flush=True)
            print(f"  predicted: factor*(Q+ΣχJ) - ΣχJ = {(factor*(total_charge[:3]+sum_chi_over_J[:3]) - sum_chi_over_J[:3]).tolist()}", flush=True)
            _os.environ['_CEP_INTERNAL_PRINTED'] = '1'

        return {
            "atomic_charges": q,
            "chi": chi,
            "J": J,
            "U_CEP_SELF": U_CEP_SELF,
            "U_CENT": U_CEP_SELF,   # alias for backward compat (Ver.1 misnomer, see module docstring)
            "total_charge": q_total,
        }
