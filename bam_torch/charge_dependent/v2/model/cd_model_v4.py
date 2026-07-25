"""
Charge-dependent model ChargeRACEv4 for BAM-torch CD Ver.2 (2026-05-21~).

Ver.2 paradigm — CENT2 FULL implementation (self + U_SLR):
  - E_total = E_SR + U_CENT  (default behavior, use_cent_energy=True)
  - U_CENT = U_CEP_SELF + U_SLR  (진짜 CENT2 paper Eq.(2) 정의)
    - U_CEP_SELF: atom self-energy (Parr-Pearson 이차, Ver.1 동일)
    - U_SLR     : pairwise shielded long-range Coulomb (Ver.2 새)
  - CEP block: hard charge conservation maintained (Lagrange analytical solution)
  - charge_type agnostic: accepts NPA / Mulliken / Hirshfeld

Differences from Ver.1 (ChargeRACEv3):
  - cep_block_v2 사용 (positions 전달, U_SLR 계산)
  - 새 hyperparameter: slr_kernel ('cent2'|'erf'|'none'), slr_lambda, slr_n, slr_sigma
  - use_cent_energy=True default (Ver.2 paradigm)
  - slr_kernel='none' 으로 Ver.1 동등 동작 가능 (ablation control)

Reference:
  Khajehpasha et al., Phys. Rev. B 105, 144106 (2022) — CENT2 (Eq.2, Eq.5)
  Ghasemi & Goedecker, J. Chem. Phys. 154, 074107 (2021) — CENT1
"""

import torch

import e3nn
from e3nn import o3

from typing import Any, Callable, Dict, List, Optional
from e3nn.util.jit import compile_mode

from bam_torch.model.blocks import (
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
    ConcatenateRaceInteractionBlock,
    RaceEquivariantBlock,
    NonLinearReadoutBlock,
    LinearForceDecoderBlock,
)
from bam_torch.model.wrapper_ops import Linear
from bam_torch.utils.scatter import scatter_sum, scatter_mean
from bam_torch.utils.output_utils import (
    get_outputs,
    get_symmetric_displacement,
    remove_net_torque,
)
from bam_torch.model.models import to_one_hot, get_edge_relative_vectors_with_pbc
from bam_torch.charge_dependent.v2.model.cep_block_v2 import CEPBlockV2


@compile_mode("script")
class ChargeRACEv4(torch.nn.Module):
    """
    ChargeRACE v4 — Ver.2 CENT2 FULL.

    Total energy decomposition:
        E_total = E_SR + U_CENT  (with use_cent_energy=True, default for Ver.2)
        U_CENT = U_CEP_SELF + U_SLR  (CENT2 paper Eq.(2))

    CEP Block v2:
        chi_i = MLP(scalar node features)      <- env-dep electronegativity
        J_i   = softplus(J_raw[species])        <- per-element hardness (learnable)
        q_i   = (lambda - chi_i) / J_i          <- Lagrange analytical solution
        U_CEP_SELF = sum_i [chi_i q_i + 0.5 J_i q_i^2]   (atom self-energy)
        U_SLR      = 0.5 sum_{i!=j} k_e q_i q_j kappa(r_ij)  (pairwise long-range)

    Args:
        charge_type    : charge type label ('npa', 'mulliken', 'hirshfeld')
        use_cent_energy: if True, E_total += U_CENT (default True for Ver.2)
        cep_hidden_dim : CEP chi_i MLP hidden dimension (default 64)
        slr_kernel     : 'cent2' | 'erf' | 'none' (default 'cent2', Ver.2 new)
        slr_lambda     : λ for cent2 kernel (default 0.340 Å^-1 = 0.18 a_0^-1, CENT2 paper optimal)
        slr_n          : n for cent2 kernel (default 4)
        slr_sigma      : σ for erf kernel (default 1.0 Å)
        slr_cutoff     : long-range cutoff (default None = all-pair within graph)
        Other args identical to ChargeRACEv3.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        avg_num_neighbors: int = 40,
        num_species: int = 1,
        max_ell: int = 3,
        num_basis_func: int = 8,
        hidden_irreps: e3nn.o3.Irreps = o3.Irreps("32x0e+32x1o+32x2e"),
        nlayers: int = 3,
        features_dim: int = 32,
        output_irreps: e3nn.o3.Irreps = o3.Irreps("1x0e"),
        active_fn: str = "swish",
        radial_MLP: Optional[List[int]] = [64, 64],
        MLP_irreps: e3nn.o3.Irreps = o3.Irreps("16x0e"),
        gate: Optional[Callable] = torch.nn.SiLU(),
        cueq_config: Optional[Dict[str, Any]] = None,
        regress_forces: str = "direct",
        compute_stress: bool = True,
        # Phase 3 parameters
        cep_hidden_dim: int = 64,
        use_cent_energy: bool = True,    # Ver.2 default = True
        charge_type: str = "npa",
        # Ver.2 (Phase 4) parameters - U_SLR
        slr_kernel: str = "cent2",        # 'cent2' | 'erf' | 'none'
        slr_lambda: float = 0.340,        # Å^-1 (= 0.18 a_0^-1, CENT2 paper optimal)
        slr_n: int = 4,                   # CENT2 paper
        slr_sigma: float = 1.0,           # Å (erf alternative)
        slr_cutoff: Optional[float] = None,
    ):
        super().__init__()

        if active_fn in ["swish", "silu", "SiLU"]:
            self.act_fn = torch.nn.SiLU()
        elif active_fn in ["relu", "ReLU"]:
            self.act_fn = torch.nn.ReLU()
        elif active_fn in ["identity", None]:
            self.act_fn = torch.nn.Identity()

        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.compute_stress = compute_stress
        self.num_species = num_species
        self.output_irreps = o3.Irreps(output_irreps)
        hidden_irreps = hidden_irreps.sort().irreps
        self.hidden_irreps = hidden_irreps
        self.nlayers = nlayers

        # Phase 3 attributes
        self.use_cent_energy: bool = use_cent_energy
        self.charge_type: str = charge_type

        # Criterion (RACE compatibility)
        self.criterion = None
        self.criterion_tag = None
        self.criterion_value = 0

        # ── 1) Embedding ──────────────────────────────────────────────────
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(features_dim, (0, 1))])
        x_node_feats_irreps = node_feats_irreps

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=1.0,
            num_bessel=num_basis_func,
            num_polynomial_cutoff=2,
            radial_type="bessel",
            distance_transform=None,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # ── 2) Interaction layers ─────────────────────────────────────────
        self.linear_x = Linear(
            x_node_feats_irreps,
            x_node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        if radial_MLP is None:
            radial_MLP = [64, 64]

        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        self.force_decoders = torch.nn.ModuleList()
        self.stress_decoders = torch.nn.ModuleList()

        target_irreps = o3.Irreps(
            f"{hidden_irreps.count(o3.Irrep(0, 1))}x0e"
        )
        for i in range(nlayers):
            if i > 0:
                node_feats_irreps = hidden_irreps
                target_irreps = hidden_irreps

            inter = ConcatenateRaceInteractionBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=target_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)

            prod = RaceEquivariantBlock(
                node_feats_irreps_1=x_node_feats_irreps,
                node_feats_irreps_2=hidden_irreps,
                output_irreps=hidden_irreps,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)

            readout = NonLinearReadoutBlock(
                irreps_in=hidden_irreps,
                MLP_irreps="64x0e",
                gate=gate,
                irrep_out=output_irreps,
                cueq_config=cueq_config,
            )
            self.readouts.append(readout)

            if "direct" in self.regress_forces:
                force_decoder = LinearForceDecoderBlock(
                    irreps_in=hidden_irreps,
                    irrep_out="1x1o",
                    cueq_config=cueq_config,
                )
                stress_decoder = LinearForceDecoderBlock(
                    irreps_in=hidden_irreps,
                    irrep_out="6x0e",
                    cueq_config=cueq_config,
                )
            else:
                force_decoder = torch.nn.Identity()
                stress_decoder = torch.nn.Identity()
            self.force_decoders.append(force_decoder)
            self.stress_decoders.append(stress_decoder)

        # ── 3) CEP Block v2 (charge predictor + U_SLR) ───────────────────
        self.cep = CEPBlockV2(
            irreps_in=hidden_irreps,
            num_species=num_species,
            hidden_dim=cep_hidden_dim,
            slr_kernel=slr_kernel,
            slr_lambda=slr_lambda,
            slr_n=slr_n,
            slr_sigma=slr_sigma,
            slr_cutoff=slr_cutoff,
        )
        # store slr_kernel for forward (needed to know if positions required)
        self._slr_kernel = slr_kernel

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        backprop: bool = False,
        compute_displacement: bool = False,
    ):
        data["cell"].requires_grad_(True)
        data["positions"].requires_grad_(True)

        displacement: Optional[torch.Tensor] = None
        if compute_displacement:
            displacement = get_symmetric_displacement(data)

        Rij = get_edge_relative_vectors_with_pbc(data)
        Rij = Rij / self.cutoff
        num_graphs = data["ptr"].numel() - 1

        # ── Atom embedding ────────────────────────────────────────────────
        if "node_attrs" in data:
            node_attrs = data["node_attrs"]
            species = data["species"]
        else:
            species = data["species"]
            node_attrs = to_one_hot(species.unsqueeze(-1), self.num_species)
        node_feats = self.node_embedding(node_attrs)

        # ── Edge embedding ────────────────────────────────────────────────
        edge_index = data["edge_index"]
        lengths = torch.norm(Rij, dim=1)

        nonzero_idx = torch.arange(
            len(lengths), device=lengths.device
        )[lengths != 0]
        Rij = Rij[nonzero_idx]
        lengths = lengths[nonzero_idx]
        edge_index = edge_index[:, nonzero_idx]

        edge_attrs = self.spherical_harmonics(Rij)
        edge_feats = self.radial_embedding(
            lengths.unsqueeze(1),
            node_attrs,
            data["edge_index"],
            species,
        )

        x_node_feats = self.linear_x(node_feats)

        frc_out = []
        sts_out = []
        outputs = []
        node_feats_list: List[torch.Tensor] = []

        # ── Interaction loop ──────────────────────────────────────────────
        for (interaction, product, readout,
             force_decoder, stress_decoder) in zip(
            self.interactions, self.products, self.readouts,
            self.force_decoders, self.stress_decoders
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                x_node_feats=x_node_feats,
                node_feats=node_feats,
                sc=sc,
            )
            node_energies = readout(node_feats)

            if "direct" in self.regress_forces:
                node_force_dir = force_decoder(node_feats)
                frc_out.append(node_force_dir)
                node_stress_dir = stress_decoder(node_feats)
                sts_out.append(node_stress_dir)

            node_feats_list.append(node_feats)
            outputs.append(node_energies[:, 0])

        # ── E_SR: short-range energy summation ────────────────────────────
        node_energy = torch.stack(outputs, dim=-1)
        node_energy = self.act_fn(node_energy)
        node_energy = torch.sum(node_energy, dim=-1)

        E_SR = scatter_sum(
            src=node_energy,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )

        # ── CEP: charges + energy term (U_CENT). Added to E_total below when
        #         use_cent_energy=True (Ver.2 default). NOTE: U_CENT enters the
        #         total energy and hence the autograd forces — it is NOT a parallel
        #         readout. (Stale "Phase 3: not included in energy" comment removed.)
        last_node_feats = node_feats_list[-1]

        if "total_charge" in data:
            total_charge = data["total_charge"].float()
            import os as _os
            if _os.environ.get('_CEP_DEBUG') and not _os.environ.get('_CEP_DEBUG_PRINTED'):
                print(f"[DEBUG] total_charge IN data: shape={tuple(total_charge.shape)} vals={total_charge[:5].tolist()} num_graphs={num_graphs}", flush=True)
                _os.environ['_CEP_DEBUG_PRINTED'] = '1'
        else:
            total_charge = torch.zeros(
                num_graphs, dtype=torch.float32, device=E_SR.device
            )
            import os as _os
            if _os.environ.get('_CEP_DEBUG') and not _os.environ.get('_CEP_DEBUG_PRINTED'):
                print(f"[DEBUG] total_charge NOT in data → zeros fallback! num_graphs={num_graphs}", flush=True)
                _os.environ['_CEP_DEBUG_PRINTED'] = '1'

        # Ver.2: CEPBlockV2 requires positions for U_SLR computation
        cep_out = self.cep(
            node_feats=last_node_feats,
            species=species,
            total_charge=total_charge,
            batch=data["batch"],
            num_graphs=num_graphs,
            positions=data["positions"],   # Ver.2 new — required for U_SLR
        )

        # ── E_total determination (Ver.2) ─────────────────────────────────
        # Ver.2 default: E_total = E_SR + U_CENT (where U_CENT = U_CEP_SELF + U_SLR)
        # use_cent_energy=False → E_total = E_SR (ablation control, ~Ver.1 behavior)
        # slr_kernel='none' → U_SLR = 0 (ablation control, Ver.1 동등)
        if self.use_cent_energy:
            graph_energy = E_SR + cep_out["U_CENT"]   # Ver.2: U_CENT = self + SLR
        else:
            graph_energy = E_SR

        preds: Dict[str, Optional[torch.Tensor]] = {}
        preds["energy"] = graph_energy
        preds["node_energy"] = node_energy
        preds["atomic_charges"] = cep_out["atomic_charges"]
        preds["total_charge"] = cep_out["total_charge"]
        preds["chi"] = cep_out["chi"]
        preds["U_CEP_SELF"] = cep_out["U_CEP_SELF"]
        preds["U_SLR"] = cep_out["U_SLR"]              # Ver.2 new
        preds["U_CENT"] = cep_out["U_CENT"]            # Ver.2: 진짜 의미 (self + SLR)
        preds["E_SR"] = E_SR

        # ── Dipole (point-charge model) ───────────────────────────────────
        # μ = Σ_i q_i (r_i − r_ref), per molecule, in e·Å.  r_ref = geometric centroid
        # (mean atomic position). For NEUTRAL molecules (Σq=0) the reference is
        # irrelevant — μ is origin-independent — so |μ| is exact for the clean case
        # (G1 neutral, G2 radical). For CHARGED molecules (G3 anion, G4 cation) μ is
        # reference-dependent AND the DFT gauge origin is unknown, so |μ| is NOT reliably
        # comparable to the DFT dipole — the evaluator stratifies and flags charged |μ|
        # as UNRELIABLE. (We do NOT use the mass-weighted COM: `species` here is the
        # remapped 0-based element index (uniq_element), NOT the atomic number, so a
        # Z-indexed mass table would gather the wrong masses.) Differentiable in q_i
        # (and pos) but NOT added to the energy → does not affect the autograd forces.
        # NOTE: only the MAGNITUDE |μ| is comparable to the QM9star DFT dipole — the
        # stored DFT dipole VECTOR is in a different (standard-orientation) frame than
        # these coordinates, so a component-wise comparison is invalid; |μ| is
        # rotation-invariant. Unit: e·Å (multiply by 4.80320 for Debye).
        _pos = data["positions"]
        _natom = scatter_sum(
            src=torch.ones_like(_pos[:, :1]), index=data["batch"], dim=0,
            dim_size=num_graphs,
        )                                                                      # (B,1)
        _centroid = scatter_sum(
            src=_pos, index=data["batch"], dim=0, dim_size=num_graphs,
        ) / _natom.clamp(min=1.0)                                              # (B,3)
        _r_rel = _pos - _centroid[data["batch"]]
        preds["dipole"] = scatter_sum(
            src=cep_out["atomic_charges"].unsqueeze(-1) * _r_rel,
            index=data["batch"], dim=0, dim_size=num_graphs,
        )

        # ── Forces ────────────────────────────────────────────────────────
        forces: Optional[torch.Tensor] = None
        stress: Optional[torch.Tensor] = None

        if self.criterion is not None:
            if self.criterion < self.criterion_value:
                self.regress_forces = "auto"
            else:
                self.regress_forces = "direct"

        if "auto" in self.regress_forces:
            forces, virials, stress, hessian = get_outputs(
                energy=graph_energy,
                positions=data["positions"],
                cell=data["cell"],
                batch_idx=data["batch"],
                num_graphs=num_graphs,
                training=backprop,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
                compute_hessian=False,
                displacement=None,
            )
            preds["forces"] = forces
            preds["stress"] = stress
            preds["virials"] = virials

        elif "direct" in self.regress_forces:
            node_force = torch.stack(frc_out, dim=-1)
            node_force = self.act_fn(node_force)
            forces = torch.sum(node_force, dim=-1)
            system_means = scatter_mean(forces, data["batch"], dim=0)
            node_broadcasted_means = system_means[data["batch"]]
            forces = forces - node_broadcasted_means
            forces = remove_net_torque(
                data["positions"], forces, data["batch"]
            )

            node_stress = torch.stack(sts_out, dim=-1)
            node_stress = self.act_fn(node_stress)
            stress = torch.sum(node_stress, dim=-1)
            stress = scatter_sum(
                src=stress,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )
            preds["forces"] = forces
            preds["stress"] = stress

        preds["displacement"] = displacement

        return preds

    def set_criterion(self, criterion_tag, criterion):
        self.criterion_tag = criterion_tag
        if "direct" in self.regress_forces:
            if criterion_tag is None:
                criterion_tag = "epoch"

        self.criterion = criterion
        if criterion_tag == "epoch":
            if criterion is None:
                self.criterion = 50
                self.criterion_value = 0
        elif criterion_tag == "loss":
            if criterion is None:
                self.criterion = 0.01
                self.criterion_value = 0.1

        self.criterion_value = 0

    def update_criterion_value(self, value):
        self.criterion_value = value
