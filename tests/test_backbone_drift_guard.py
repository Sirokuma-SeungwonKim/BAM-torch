"""
Drift-guard regression test: RACE backbone  ==  ChargeRACEv4 backbone (E_SR + forces).

WHY (2026-05-29 audit):
    ChargeRACEv4 (CD model, stages S1-S4) does NOT inherit from RACE (baseline S0).
    It COPY-PASTES the short-range backbone (embedding -> interaction/product/readout
    loop -> E_SR pooling) and bolts on CEPBlockV2. The paper's controlled comparison
    assumes S0 and the CD models share an *identical* short-range backbone. Because the
    backbone is copied (not inherited / not composed), a future edit to RACE will NOT
    propagate to ChargeRACEv4, and the two can silently DRIFT — invalidating the
    S0-vs-CD comparison without any error being raised.

    This test locks in the verified-equivalent state found in the audit:

      [STRUCTURAL] every RACE backbone parameter key exists in ChargeRACEv4 with a
                   matching shape  (the module structure has not diverged).
      [NUMERICAL]  with use_cent_energy=False (E_total = E_SR, no U_CENT) and the
                   backbone weights copied CD <- RACE, ChargeRACEv4's E_SR and forces
                   are bit-equal to RACE's energy and forces on a fixed molecular batch.

    If this test FAILS, the copy-pasted backbones have drifted. Either re-sync them or
    re-run the audit before trusting any S0-vs-CD result.

    NOTE: this is the runtime confirmation of a claim that was previously only verified
    at the source level (RACE.forward vs ChargeRACEv4.forward line-by-line diff).

Run:  python tests/test_backbone_drift_guard.py    (also importable as a pytest test)
"""
import numpy as np
import torch
from e3nn import o3
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.data import Batch

from bam_torch.model.models import RACE
from bam_torch.charge_dependent.v2.model.cd_model_v4 import ChargeRACEv4
from bam_torch.charge_dependent.utils.cd_utils import get_graphset_charge

# ── Shared backbone config (matches the phase9v2 run configs) ──────────────
CUTOFF = 5.0
UNIQ_ELEMENT = {1: 0, 6: 1, 7: 2, 8: 3}   # H, C, N, O
NUM_SPECIES = 4
HIDDEN_IRREPS = o3.Irreps("32x0e+32x1o+32x2e")

BACKBONE_KW = dict(
    cutoff=CUTOFF,
    avg_num_neighbors=16,
    num_species=NUM_SPECIES,
    max_ell=2,
    num_basis_func=8,
    hidden_irreps=HIDDEN_IRREPS,
    nlayers=3,
    features_dim=64,
    output_irreps=o3.Irreps("1x0e"),
    active_fn="swish",
    radial_MLP=[64, 64],
    MLP_irreps=o3.Irreps("16x0e"),
    regress_forces="auto",     # -> autograd forces (= the actual run setting)
    compute_stress=True,
)


def _build_atoms():
    """Three small neutral molecules (H2O, CH4, NH3) with dummy energy/forces/charges."""
    mols = [
        (["O", "H", "H"],
         [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]],
         [-0.82, 0.41, 0.41]),
        (["C", "H", "H", "H", "H"],
         [[0, 0, 0], [0.628, 0.628, 0.628], [-0.628, -0.628, 0.628],
          [-0.628, 0.628, -0.628], [0.628, -0.628, -0.628]],
         [-0.64, 0.16, 0.16, 0.16, 0.16]),
        (["N", "H", "H", "H"],
         [[0, 0, 0.116], [0, 0.938, -0.271],
          [0.812, -0.469, -0.271], [-0.812, -0.469, -0.271]],
         [-1.02, 0.34, 0.34, 0.34]),
    ]
    rng = np.random.RandomState(0)
    out = []
    for syms, crd, chg in mols:
        a = Atoms(symbols=syms, positions=np.array(crd, dtype=float),
                  cell=np.diag([30., 30., 30.]), pbc=False)
        a.calc = SinglePointCalculator(
            a, energy=float(rng.normal(-50, 1)),
            forces=rng.normal(0, 0.01, (len(syms), 3)))
        a.arrays["charges"] = np.array(chg, dtype=float)
        a.info["total_charge"] = 0.0
        out.append(a)
    return out


def _make_batch():
    """Build one PyG batch via the exact CD data pipeline (variable-size, no padding)."""
    graphs = get_graphset_charge(
        _build_atoms(), CUTOFF, UNIQ_ELEMENT,
        enr_avg_per_element=np.zeros(NUM_SPECIES), enr_var=1.0,
        regress_forces=True)
    return Batch.from_data_list(graphs)


def _build_models():
    torch.manual_seed(0)
    race = RACE(**BACKBONE_KW)
    race.criterion = None          # RACE sets this only in set_criterion(); init it here
    race.criterion_value = 0
    torch.manual_seed(0)
    cd = ChargeRACEv4(**BACKBONE_KW, cep_hidden_dim=64,
                      use_cent_energy=False, charge_type="npa", slr_kernel="none")
    return race, cd


def test_backbone_drift_guard():
    race, cd = _build_models()

    # sanity: backbone hyperparams resolved identically
    assert str(race.hidden_irreps) == str(cd.hidden_irreps), \
        f"hidden_irreps diverged: {race.hidden_irreps} vs {cd.hidden_irreps}"

    race_sd, cd_sd = race.state_dict(), cd.state_dict()

    # ── [STRUCTURAL] every RACE backbone key present in CD with same shape ──
    missing = [k for k in race_sd if k not in cd_sd]
    mismatched = [k for k in race_sd
                  if k in cd_sd and tuple(cd_sd[k].shape) != tuple(race_sd[k].shape)]
    assert not missing, \
        f"STRUCTURAL DRIFT: RACE backbone keys absent from ChargeRACEv4: {missing}"
    assert not mismatched, \
        f"STRUCTURAL DRIFT: shape mismatch on shared backbone keys: {mismatched}"

    # copy backbone weights CD <- RACE (keep CD's own cep.* params)
    cd.load_state_dict({**cd_sd, **{k: race_sd[k] for k in race_sd}}, strict=True)

    # ── [NUMERICAL] E_SR and forces must be identical on the same batch ──
    race.eval()
    cd.eval()
    pr = race(_make_batch(), backprop=True)
    pc = cd(_make_batch(), backprop=True)

    e_race, e_cd = pr["energy"], pc["E_SR"]   # RACE energy == its E_SR (no CEP)
    de = (e_race - e_cd).abs().max().item()
    assert torch.allclose(e_race, e_cd, atol=1e-5, rtol=1e-4), \
        f"NUMERICAL DRIFT (E_SR): max|Δ|={de:.3e}\n RACE={e_race.tolist()}\n CD  ={e_cd.tolist()}"

    f_race, f_cd = pr["forces"], pc["forces"]
    df = (f_race - f_cd).abs().max().item()
    assert torch.allclose(f_race, f_cd, atol=1e-5, rtol=1e-4), \
        f"NUMERICAL DRIFT (forces): max|Δ|={df:.3e}"

    print("[PASS] backbone drift-guard: RACE backbone == ChargeRACEv4 backbone")
    print(f"       shared backbone keys : {len(race_sd)}  (CD-only cep keys: {len(cd_sd) - len(race_sd)})")
    print(f"       E_SR  : RACE={[round(x,6) for x in e_race.tolist()]}  max|Δ|={de:.2e}")
    print(f"       forces: max|ΔF|={df:.2e}")
    return True


if __name__ == "__main__":
    test_backbone_drift_guard()
