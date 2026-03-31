# bam_torch.charge_dependent — Package Overview

## What is this package?

The `charge_dependent` sub-package extends BAM-torch with **charge-aware
machine-learning interatomic potentials**.  It adds a Charge Equilibration
Process (CEP) block on top of the RACE equivariant graph neural network so
that the model can simultaneously predict **energies, forces, and atomic
partial charges** in a single forward pass.

Charge conservation (sum of atomic charges = total system charge) is
enforced **exactly** via an analytical Lagrange-multiplier solution — no
soft penalty is needed.

---

## Package structure

```
charge_dependent/
├── __init__.py
├── PACKAGE_OVERVIEW.md        ← this file
├── USAGE_GUIDE.md             ← usage examples
├── model/
│   ├── __init__.py            # MODEL_REGISTRY (charge_race, charge_race_v3, …)
│   ├── cep_block.py           # CEPBlock — CENT2-based charge equilibration
│   ├── cd_model.py            # ChargeRACE   (Phase 2: E = E_SR + U_CENT)
│   └── cd_model_v3.py         # ChargeRACEv3 (Phase 3: E = E_SR only)
├── training/
│   ├── __init__.py            # TRAINER_REGISTRY (cd, cd_v3, …)
│   ├── cd_trainer.py          # CDTrainer   (Phase 2)
│   └── cd_trainer_v3.py       # CDTrainerV3 (Phase 3)
├── predicting/
│   ├── __init__.py            # EVALUATOR_REGISTRY (cd, cd_v3, …)
│   ├── cd_evaluator.py        # CDEvaluator   (Phase 2)
│   └── cd_evaluator_v3.py     # CDEvaluatorV3 (Phase 3)
├── calculator/
│   ├── __init__.py
│   └── cd_calculator.py       # CDRACECalculator — ASE Calculator with ZBL prior
└── utils/
    ├── cd_utils.py            # Data loading, graph conversion (DDP-aware)
    └── qm9star_preprocessor.py  # QM9star SQL dump → extended XYZ converter
```

---

## Model variants

| Key | Model class | Energy formula | Charge conservation | When to use |
|-----|-------------|----------------|---------------------|-------------|
| `charge_race` | ChargeRACE | E_SR + U_CENT | Hard (Lagrange) | When electrostatic energy matters |
| `charge_race_v3` | ChargeRACEv3 | **E_SR only** | Hard (Lagrange) | General-purpose (recommended) |

### CEP Block (shared by both variants)

The Charge Equilibration Process block predicts atomic charges analytically:

1. **chi_i** = MLP(scalar node features) — environment-dependent electronegativity
2. **J_i** = softplus(J_raw[species]) — per-element chemical hardness (learnable)
3. **lambda** = (Q_total + sum(chi/J)) / sum(1/J) — Lagrange multiplier (per graph)
4. **q_i** = (lambda - chi_i) / J_i — atomic charges (hard conservation guaranteed)

Reference: Khajehpasha et al., Phys. Rev. B 105, 144106 (2022) — CENT2

---

## Trainer variants

| Key | Trainer class | Parent | Notes |
|-----|---------------|--------|-------|
| `cd` / `charge_dependent` | CDTrainer | BaseTrainer | Phase 2 full trainer |
| `cd_v3` / `cd_e` | CDTrainerV3 | CDTrainer | Phase 3, overrides `set_model()` only |

### Loss function

```
total_loss = enr_lambda * loss_E  +  frc_lambda * loss_F  +  chg_lambda * loss_Q
```

- `loss_E`: energy MSE/RMSE (per graph)
- `loss_F`: force MSE/RMSE (per atom per component)
- `loss_Q`: charge MSE (predicted q_i vs reference charges)

---

## Multi-GPU (DDP) support

`cd_utils.get_dataloader_charge()` automatically wraps DataLoaders with
`DistributedSampler` when `world_size > 1`, enabling multi-GPU training
via PyTorch DDP without code changes.

---

## ASE Calculator

`CDRACECalculator` inherits from `RACECalculator` and adds:

- Charge prediction (`results['charges']`)
- ZBL repulsive prior for MD stability (prevents unphysical atom overlap)
- Configurable via the same JSON config used for training

---

## Registry keys reference

### Models
- `charge_race`, `cd_race` → ChargeRACE (Phase 2)
- `charge_race_v3`, `cd_race_v3`, `charge_race_e` → ChargeRACEv3 (Phase 3)

### Trainers
- `cd`, `charge_dependent` → CDTrainer (Phase 2)
- `cd_v3`, `charge_dependent_v3`, `cd_e` → CDTrainerV3 (Phase 3)

### Evaluators
- `cd` → CDEvaluator (Phase 2)
- `cd_v3`, `cd_e` → CDEvaluatorV3 (Phase 3)
