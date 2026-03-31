# bam_torch.charge_dependent — Usage Guide

## Prerequisites

```bash
# Required packages
conda activate bam_torch  # or your environment name
pip install torch torch_geometric e3nn ase matscipy tqdm

# For GPU training
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

---

## 1. Data preparation

### Convert QM9star SQL dump to extended XYZ

```bash
python -m bam_torch.charge_dependent.utils.qm9star_preprocessor \
    --sql qm9star_plain.sql \
    --charge-type npa_charges \
    --energy-type U_0 \
    --max-samples 10000 \
    --output qm9star_10000.xyz
```

The output extended XYZ has the format:
```
9
Lattice="30.0 0.0 0.0 ..." Properties=species:S:1:pos:R:3:forces:R:3:charges:R:1 energy=-2120.5 total_charge=0.0 pbc="F F F"
C    0.00000000   0.00000000   0.00000000   0.00100000   0.00200000  -0.00300000    -0.123456
H    1.09000000   0.00000000   0.00000000  -0.00050000   0.00010000   0.00020000     0.045678
...
```

### Supported charge types
- `npa_charges` (Natural Population Analysis) — recommended
- `mulliken_charge`
- `hirshfeld_charges`
- `formal_charges`

---

## 2. Training

### Minimal config (Phase 3, recommended)

Create `input.json`:

```json
{
    "device": "gpu",
    "model": "charge_race_v3",
    "trainer": "cd_v3",
    "cueq_config": false,
    "regress_forces": true,

    "fname_traj": "qm9star_10000.xyz",
    "ntrain": 8000,
    "nvalid": 2000,
    "element": "auto",
    "cutoff": 6.0,
    "avg_num_neighbors": 30,
    "num_species": 5,
    "max_ell": 2,
    "num_radial_basis": 8,
    "hidden_channels": "64x0e+32x1o+16x2e",
    "output_channels": "1x0e",
    "nbatch": 32,
    "nlayers": 4,
    "features_dim": 128,
    "active_fn": "identity",
    "pbc": false,

    "charge": {
        "cep_hidden_dim": 64,
        "use_cent_energy": false,
        "charge_type": "npa",
        "charge_key": "charges",
        "total_charge_key": "total_charge",
        "charge_loss": "mse"
    },

    "NN": {
        "data_seed": 10,
        "init_seed": 11,
        "learning_rate": 0.005,
        "weight_decay": 1e-5,
        "nepoch": 300,
        "nsave": 60,
        "restart": false,
        "fname_pkl": "model.pkl",
        "loss_config": {
            "energy_loss": "mse",
            "force_loss": "mse"
        },
        "enr_lambda": 1,
        "frc_lambda": 50,
        "chg_lambda": 5,
        "l2_lambda": 0.0
    },

    "scheduler": {
        "scheduler": "CosineAnnealingWarmRestarts",
        "T_0": 50,
        "T_mult": 2,
        "eta_min": 1e-6
    },

    "log_length": "simple",
    "log_interval": 5,
    "log_config": {
        "step": ["date", "epoch"],
        "train": ["loss", "loss_e", "loss_f", "loss_q"],
        "valid": ["loss", "loss_e", "loss_f", "loss_q"],
        "lr": ["lr"]
    },
    "train": {
        "fname_log": "loss_train.out"
    }
}
```

### Run training

```bash
python -m bam_torch.training.run_train --config input.json
```

### Key parameters to tune

| Parameter | Description | Recommended range |
|-----------|-------------|-------------------|
| `frc_lambda` | Force loss weight | 25–50 (dataset-dependent) |
| `chg_lambda` | Charge loss weight | 5–10 |
| `nbatch` | Batch size | 32 (local GPU) / 64–128 (server) |
| `hidden_channels` | Model capacity | `"64x0e+32x1o+16x2e"` for ~3M params |
| `nepoch` | Training epochs | 300 (quick) / 5000 (production) |

---

## 3. Evaluation

### Config for prediction

Add a `predict` section to `input.json` (or use a separate config):

```json
{
    "predict": {
        "evaluate_tag": false,
        "loss_config": {
            "energy_loss": "rmse",
            "force_loss": "rmse"
        },
        "fname_traj": "test_data.xyz",
        "ndata": "test_data.xyz",
        "model": "model.pkl",
        "fname_plog": "predict.out"
    }
}
```

### Run evaluation

```bash
python -m bam_torch.predicting.run_predict --config input.json
```

This outputs `test_values.pkl` containing predicted and reference values
for energy, forces, and charges.

---

## 4. ASE Calculator (MD simulations)

```python
from ase.io import read
from ase.md.langevin import Langevin
from ase import units

from bam_torch.charge_dependent.calculator import CDRACECalculator

# Load trained model
calc = CDRACECalculator(model="model.pkl", device="cuda")

# Attach to atoms
atoms = read("structure.xyz")
atoms.calc = calc

# Single-point calculation
energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV/Ang
charges = calc.results["charges"]       # elementary charge (e)

# Langevin MD at 300 K
dyn = Langevin(atoms, timestep=1.0 * units.fs,
               temperature_K=300,
               friction=0.01 / units.fs)  # note: ASE internal units

dyn.run(steps=1000)
```

---

## 5. Multi-GPU training (DDP)

### Launcher script

```python
# main_cd.py
import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    with open("input.json") as f:
        config = json.load(f)

    from bam_torch.charge_dependent.training import TRAINER_REGISTRY
    trainer_cls = TRAINER_REGISTRY[config["trainer"]]
    trainer = trainer_cls(config, rank=rank, world_size=world_size)
    trainer.run()

if __name__ == "__main__":
    main()
```

### SLURM job script

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00

export CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun --nproc_per_node=4 main_cd.py
```

---

## 6. Resuming from checkpoint

Set `restart` to `true` in the config and ensure `fname_pkl` points to the
saved checkpoint:

```json
{
    "NN": {
        "restart": true,
        "fname_pkl": "model.pkl"
    }
}
```

Then re-run the same training command.

---

## Quick reference: model/trainer/evaluator keys

| Use case | `"model"` | `"trainer"` | Evaluator |
|----------|-----------|-------------|-----------|
| Phase 2 (E_SR + U_CENT) | `"charge_race"` | `"cd"` | CDEvaluator |
| **Phase 3 (E_SR only)** | `"charge_race_v3"` | `"cd_v3"` | CDEvaluatorV3 |
