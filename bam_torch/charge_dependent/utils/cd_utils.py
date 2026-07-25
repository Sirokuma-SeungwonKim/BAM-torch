"""
Charge-dependent data utilities for BAM-torch.

Utilities for converting QM9star data into PyG Data objects with charge information.
Also supports Wiggle150 test set loading.

Data sources:
  - Extended xyz files converted by qm9star_preprocessor.py
  - Or general xyz/traj files containing charge info
"""

import os
import gc
import random
import hashlib
import numpy as np
from ase.io import read
from matscipy.neighbours import neighbour_list
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from bam_torch.utils.utils import (
    get_enr_avg_per_element,
    get_relative_vector,
)

# =============================================================
# Unit conversion constants
# =============================================================
HARTREE_TO_EV = 27.211386245988       # eV/Hartree
BOHR_TO_ANGSTROM = 0.529177249        # Ang/bohr
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # ~ 51.4221


def _safe_get_forces(atoms):
    """Safely retrieve forces from an atoms object.

    In ASE 3.27+, 'forces' is stored in the internal calculator, not in arrays.
    """
    # (1) Read directly from arrays (older ASE)
    if 'forces' in atoms.arrays:
        return atoms.arrays['forces']
    # (2) ASE 3.27+: call get_forces() directly
    try:
        f = atoms.get_forces()
        if f is not None:
            return f
    except Exception:
        pass
    # (3) Read from calculator results
    if atoms.calc is not None:
        try:
            return atoms.get_forces()
        except Exception:
            pass
    return np.zeros((len(atoms), 3))


def _safe_get_energy(atoms):
    """Safely retrieve energy from an atoms object.

    In ASE 3.27+, 'energy' is accessed via get_potential_energy(), not info dict.
    """
    # (1) Read directly from info (older ASE)
    if 'energy' in atoms.info:
        return float(atoms.info['energy'])
    # (2) ASE 3.27+: call get_potential_energy() directly
    try:
        e = atoms.get_potential_energy()
        if e is not None:
            return float(e)
    except Exception:
        pass
    # (3) Read from calculator results
    if atoms.calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    raise ValueError("Could not find energy. "
                     "Extended xyz or calculator is required.")


def _safe_get_stress(atoms):
    """Safely retrieve stress from an atoms object."""
    if atoms.calc is not None:
        try:
            if 'stress' in atoms.calc.results:
                return atoms.get_stress()
        except Exception:
            pass
    return np.zeros(6)


def _safe_get_volume(atoms):
    """Safely retrieve volume from an atoms object."""
    try:
        vol = atoms.get_volume()
        if vol > 0:
            return vol
    except Exception:
        pass
    return np.zeros(1)


def _extract_charges(atoms, charge_key="charges",
                     total_charge_key="total_charge"):
    """Extract charge information from an atoms object.

    In extended xyz format:
      - atoms.arrays['charges'] -> per-atom charges
      - atoms.info['total_charge'] -> system total charge

    Returns:
        atomic_charges: np.ndarray [n_atoms]
        total_charge: float
    """
    n_atoms = len(atoms)

    # --- Atomic charges ---
    atomic_charges = None

    # (1) Find in arrays (older ASE: Properties=...charges:R:1)
    if charge_key in atoms.arrays:
        atomic_charges = np.array(atoms.arrays[charge_key], dtype=float)
    # (1b) ASE 3.27+: 'charges' accessed via get_charges()
    elif charge_key == 'charges':
        try:
            q = atoms.get_charges()
            if q is not None and len(q) == n_atoms:
                atomic_charges = np.array(q, dtype=float)
        except Exception:
            pass
    # (2) Find in info
    elif charge_key in atoms.info:
        val = atoms.info[charge_key]
        if isinstance(val, (list, np.ndarray)):
            atomic_charges = np.array(val, dtype=float)
        elif isinstance(val, str):
            atomic_charges = np.array(
                [float(x) for x in val.split()], dtype=float
            )

    # (3) Try other common key names
    if atomic_charges is None:
        for alt_key in ['charge', 'npa_charges', 'mulliken_charge',
                        'hirshfeld_charges', 'formal_charges',
                        'initial_charges']:
            if alt_key in atoms.arrays:
                atomic_charges = np.array(
                    atoms.arrays[alt_key], dtype=float
                )
                break
            elif alt_key in atoms.info:
                val = atoms.info[alt_key]
                if isinstance(val, (list, np.ndarray)):
                    atomic_charges = np.array(val, dtype=float)
                    break

    if atomic_charges is None:
        atomic_charges = np.zeros(n_atoms)

    # --- Total charge ---
    if total_charge_key in atoms.info:
        total_charge = float(atoms.info[total_charge_key])
    else:
        total_charge = float(np.sum(atomic_charges))

    # --- Total multiplicity ---
    mult_key = "total_multiplicity"
    if mult_key in atoms.info:
        total_multiplicity = int(atoms.info[mult_key])
    else:
        total_multiplicity = 1  # default: singlet

    return atomic_charges, total_charge, total_multiplicity


def _extract_observables(atoms):
    """Extract per-structure electrostatic observables (dipole, quadrupole).

    Optional supervision targets (AIMNet2 P1 / gauge-depth). Absent targets are
    returned as zeros together with a presence flag so a downstream loss can mask
    them out — a stored zero must never be trained on as if it were real.

    Sources handled:
      - dipole:     info['dipole'] (3-vector)  OR  info['dipole_x/y/z']  (QM9star)
      - quadrupole: info['quadrupole'] (6-vector: xx,yy,zz,xy,xz,yz — AIMNet2 order)
                    or a full 3x3 (9) collapsed to the 6 unique components.

    Returns:
        dipole:     np.ndarray [3]   (zeros if absent)
        quadrupole: np.ndarray [6]   (zeros if absent)
        has_dipole:     float 1.0/0.0
        has_quadrupole: float 1.0/0.0
    """
    info = atoms.info

    # --- dipole ---
    dipole = None
    if 'dipole' in info:
        v = np.asarray(info['dipole'], dtype=float).ravel()
        if v.size == 3:
            dipole = v
    if dipole is None and all(f'dipole_{ax}' in info for ax in 'xyz'):
        dipole = np.array([float(info['dipole_x']),
                           float(info['dipole_y']),
                           float(info['dipole_z'])])
    has_dipole = dipole is not None
    if dipole is None:
        dipole = np.zeros(3)

    # --- quadrupole (6 unique components) ---
    quad = None
    if 'quadrupole' in info:
        v = np.asarray(info['quadrupole'], dtype=float).ravel()
        if v.size == 6:
            quad = v
        elif v.size == 9:  # full 3x3 -> [xx,yy,zz,xy,xz,yz]
            m = v.reshape(3, 3)
            quad = np.array([m[0, 0], m[1, 1], m[2, 2],
                             m[0, 1], m[0, 2], m[1, 2]])
    has_quadrupole = quad is not None
    if quad is None:
        quad = np.zeros(6)

    return dipole, quad, float(has_dipole), float(has_quadrupole)


def get_graphset_charge(
    data, cutoff, uniq_element, enr_avg_per_element,
    enr_var, regress_forces=True, max_neigh=None,
    show_progress=False, desc="Converting",
    charge_key="charges",
    total_charge_key="total_charge",
):
    """
    Create graph dataset with charge information (extension of get_graphset).

    Supports both extended xyz (qm9star_preprocessor.py output) and general ASE files.

    Args:
        data: list of ASE atoms objects
        cutoff: distance cutoff (Angstrom)
        uniq_element: {atomic_number: species_index} dictionary
        enr_avg_per_element: per-element average energy
        enr_var: energy variance
        regress_forces: whether to regress forces
        max_neigh: maximum number of neighbors
        show_progress: whether to show tqdm progress bar
        desc: progress bar description
        charge_key: atomic charge key name
        total_charge_key: total charge key name

    Returns:
        graph_list: list of PyG Data objects
    """
    graph_list = []
    iterator = tqdm(data, desc=desc, leave=False) if show_progress else data

    for atoms in iterator:
        crds = atoms.get_positions()
        node_enr_avg = np.array([
            enr_avg_per_element[uniq_element[iz]]
            for iz in atoms.numbers
        ])

        # Energy (safe read)
        enr = _safe_get_energy(atoms) - node_enr_avg.sum()

        # Forces (safe read)
        if regress_forces or regress_forces == 'direct':
            frc = _safe_get_forces(atoms)
            volume = _safe_get_volume(atoms)
        else:
            frc = np.zeros((len(atoms), 3))
            volume = np.zeros(1)

        # Cell handling
        cell = np.array(atoms.get_cell())
        if np.all(cell == 0.0):
            cell = np.diag([30., 30., 30.])
            atoms.set_cell(cell)

        # Stress
        stress = _safe_get_stress(atoms)
        if np.all(stress == 0):
            volume = np.zeros(1)

        # Neighbor list
        iatoms, jatoms, Sij = neighbour_list(
            quantities='ijS', atoms=atoms, cutoff=cutoff
        )
        species = np.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        # Maximum neighbor count limit
        if max_neigh is not None:
            Rij, dist = get_relative_vector(atoms, iatoms, jatoms, Sij)
            nonmax_idx = []
            for i in range(len(atoms)):
                idx_i = (iatoms == i).nonzero()[0]
                idx_sorted = np.argsort(dist[idx_i])[:max_neigh]
                nonmax_idx.append(idx_i[idx_sorted])
            nonmax_idx = np.concatenate(nonmax_idx)
            iatoms = iatoms[nonmax_idx]
            jatoms = jatoms[nonmax_idx]
            num_edges = iatoms.shape[0]
            Sij = Sij[nonmax_idx]

        # Extract charge information
        atomic_charges, total_charge, total_multiplicity = _extract_charges(
            atoms, charge_key, total_charge_key
        )

        # Extract optional observable targets (dipole/quadrupole) + masks
        dipole, quadrupole, has_dipole, has_quadrupole = _extract_observables(
            atoms
        )

        # Create PyG Data object
        graph = Data(
            positions=torch.tensor(crds, dtype=torch.float32),
            species=torch.tensor(species, dtype=torch.long),
            forces=torch.tensor(frc, dtype=torch.float32),
            edges=torch.tensor(Sij, dtype=torch.float32),
            num_nodes=num_nodes,
            num_edges=num_edges,
            energy=torch.tensor(enr, dtype=torch.float32),
            cell=torch.tensor(
                np.array(cell), dtype=torch.float32
            ).view(1, 3, 3),
            edge_index=torch.tensor(
                np.array([iatoms, jatoms]), dtype=torch.long
            ),
            stress=torch.tensor(stress, dtype=torch.float32),
            volume=torch.tensor(volume),
            # Charge-related fields
            atomic_charges=torch.tensor(
                atomic_charges, dtype=torch.float32
            ),
            total_charge=torch.tensor(
                total_charge, dtype=torch.float32
            ),
            total_multiplicity=torch.tensor(
                total_multiplicity, dtype=torch.long
            ),
            # Optional observable targets (graph-level; mirror `cell`'s (1,...)
            # leading dim so PyG batches them to (B,3)/(B,6)). has_* = mask.
            dipole=torch.tensor(dipole, dtype=torch.float32).view(1, 3),
            quadrupole=torch.tensor(quadrupole, dtype=torch.float32).view(1, 6),
            has_dipole=torch.tensor([has_dipole], dtype=torch.float32),
            has_quadrupole=torch.tensor([has_quadrupole], dtype=torch.float32),
        )
        graph_list.append(graph)

    return graph_list


def _get_cache_path(fname, ntrain, nvalid, random_seed, cutoff,
                    charge_key, regress_forces, max_neigh):
    """Generate deterministic cache file path based on data parameters."""
    fsize = os.path.getsize(fname)
    key = (f"{os.path.abspath(fname)}|{fsize}|{ntrain}|{nvalid}|"
           f"{random_seed}|{cutoff}|{charge_key}|{regress_forces}|{max_neigh}")
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    basename = os.path.splitext(os.path.basename(fname))[0]
    cache_dir = os.path.dirname(os.path.abspath(fname))
    return os.path.join(cache_dir, f".cache_{basename}_{h}.pt")


def _wait_for_cache(cache_path, rank, poll_interval=10):
    """Wait for rank 0 to finish writing the cache file.

    Uses file-based synchronization instead of dist.barrier() to avoid
    NCCL timeout when rank 0 takes a long time to convert large datasets.
    """
    import time
    ready_path = cache_path + '.ready'
    if rank == 0:
        # Signal that cache is ready
        with open(ready_path, 'w') as f:
            f.write('ready')
    else:
        # Poll for the sentinel file
        while not os.path.exists(ready_path):
            time.sleep(poll_interval)
        time.sleep(1)  # ensure file is fully flushed


class ShardedDataset(torch.utils.data.IterableDataset):
    """Streams PyG graphs from pre-built shard .pt files one shard at a time.

    Peak RAM = 1 shard's worth of data, regardless of total dataset size.
    Each epoch the shard order and graph order within each shard are reshuffled
    using (seed + iter_count) so every epoch sees a different order.

    Supports multi-worker DataLoader: shards are split across workers via
    get_worker_info() so no shard is loaded twice.
    """

    def __init__(self, shard_files, cache_dir, shuffle=True, seed=42):
        """
        Args:
            shard_files: list of (filename, n_graphs) tuples for THIS rank
            cache_dir: directory containing the shard .pt files
            shuffle: whether to shuffle shard order and graphs within each shard
            seed: base random seed (each iteration uses seed + iter_count)
        """
        self.shard_files = list(shard_files)
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.seed = seed
        self._iter_count = 0
        self._n_graphs = sum(count for _, count in self.shard_files)

    def __len__(self):
        return self._n_graphs

    def __iter__(self):
        self._iter_count += 1
        rng = random.Random(self.seed + self._iter_count)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split shards across workers: worker k handles shards k, k+W, k+2W, ...
            my_shards = self.shard_files[worker_info.id::worker_info.num_workers]
        else:
            my_shards = self.shard_files

        my_shards = list(my_shards)
        if self.shuffle:
            rng.shuffle(my_shards)

        for fname, _count in my_shards:
            shard = torch.load(os.path.join(self.cache_dir, fname),
                               weights_only=False)
            if self.shuffle:
                rng.shuffle(shard)
            yield from shard
            del shard
            gc.collect()


def _load_sharded_cache(cache_dir, manifest_name='manifest_cd.pt',
                         rank=0, world_size=1):
    """Load pre-built sharded cache from manifest + shard files.

    Each rank loads only its interleaved subset of shards (rank, rank+world_size,
    rank+2*world_size, ...) so peak RAM scales as 1/world_size instead of
    loading the full dataset on every rank.

    Args:
        cache_dir: directory containing manifest and shard .pt files
        manifest_name: manifest filename (default: manifest_cd.pt)
        rank: DDP rank (default 0)
        world_size: DDP world size (default 1)

    Returns:
        train_graphs, valid_graphs, uniq_element, enr_avg_per_element
    """
    manifest_path = os.path.join(cache_dir, manifest_name)
    manifest = torch.load(manifest_path, weights_only=False)

    all_train_shards = manifest['train_shards']
    all_valid_shards = manifest['valid_shards']

    # Each rank gets floor(n_shards / world_size) shards contiguously.
    # Leftover shards are intentionally dropped (< world_size shards = < 1 epoch's
    # worth, negligible for large datasets). This guarantees all ranks have the
    # same number of shards → equal DataLoader length → no DDP hang.
    n_train_per_rank = len(all_train_shards) // world_size
    n_valid_per_rank = max(1, len(all_valid_shards) // world_size)
    my_train_shards = all_train_shards[rank * n_train_per_rank:
                                        (rank + 1) * n_train_per_rank]
    my_valid_shards = all_valid_shards[rank * n_valid_per_rank:
                                        (rank + 1) * n_valid_per_rank]

    if rank == 0:
        n_ignored_train = len(all_train_shards) - n_train_per_rank * world_size
        n_ignored_valid = len(all_valid_shards) - n_valid_per_rank * world_size
        print(f"\033[32mLoading sharded cache: {cache_dir}\033[0m")
        print(f"  Train total: {manifest['n_train']:,} graphs in "
              f"{len(all_train_shards)} shards → "
              f"{n_train_per_rank} shards/rank "
              f"({n_ignored_train} shard(s) dropped for balance)")
        print(f"  Valid total: {manifest['n_valid']:,} graphs in "
              f"{len(all_valid_shards)} shards → "
              f"{n_valid_per_rank} shards/rank "
              f"({n_ignored_valid} shard(s) dropped for balance)")

    train_graphs = []
    for fname_shard, count in tqdm(my_train_shards,
                                   desc=f"[Rank {rank}] Loading train shards"):
        shard = torch.load(os.path.join(cache_dir, fname_shard),
                           weights_only=False)
        train_graphs.extend(shard)
        del shard

    valid_graphs = []
    for fname_shard, count in tqdm(my_valid_shards,
                                   desc=f"[Rank {rank}] Loading valid shards"):
        shard = torch.load(os.path.join(cache_dir, fname_shard),
                           weights_only=False)
        valid_graphs.extend(shard)
        del shard

    if rank == 0:
        print(f"  [Rank 0] Loaded: {len(train_graphs):,} train / "
              f"{len(valid_graphs):,} valid graphs")

    return (train_graphs, valid_graphs,
            manifest['uniq_element'], manifest['enr_avg_per_element'])


def get_dataloader_charge(
    fname, ntrain, nvalid, nbatch, cutoff, random_seed,
    element=None, regress_forces=True, max_neigh=None,
    charge_key="charges", total_charge_key="total_charge",
    rank=0, world_size=1, cache_dir=None,
):
    """
    Create DataLoader with charge information.

    Accepts extended xyz converted by qm9star_preprocessor.py or
    general ASE files as input.

    For DDP (world_size > 1), only rank 0 reads and converts the raw data,
    then saves to a .pt cache file. Other ranks load from the cache after
    a barrier, reducing peak CPU memory from N×full to 1×full + N×graphs.

    Args:
        fname: structure file path (.xyz, .traj, etc.)
        ntrain: number of training samples
        nvalid: number of validation samples
        nbatch: batch size
        cutoff: distance cutoff
        random_seed: random seed
        element: element list ("auto" or list)
        regress_forces: whether to regress forces
        max_neigh: maximum number of neighbors
        charge_key: atomic charge key name
        total_charge_key: total charge key name
        rank: DDP rank
        world_size: DDP world_size
        cache_dir: pre-built sharded cache directory (overrides XYZ conversion)

    Returns:
        train_loader, valid_loader, uniq_element, enr_avg_per_element
    """
    # --- Pre-built sharded cache: streaming shard-by-shard loading ---
    if cache_dir is not None:
        manifest_path = os.path.join(cache_dir, 'manifest_cd.pt')
        if os.path.exists(manifest_path):
            manifest = torch.load(manifest_path, weights_only=False)
            all_train_shards = manifest['train_shards']
            all_valid_shards = manifest['valid_shards']
            uniq_element = manifest['uniq_element']
            enr_avg_per_element = manifest['enr_avg_per_element']

            # Each rank gets floor(n_shards / world_size) shards contiguously.
            # Leftover shards are dropped to keep all ranks equal (DDP safety).
            n_train_per_rank = len(all_train_shards) // world_size
            n_valid_per_rank = max(1, len(all_valid_shards) // world_size)
            my_train_shards = all_train_shards[
                rank * n_train_per_rank : (rank + 1) * n_train_per_rank
            ]
            my_valid_shards = all_valid_shards[
                rank * n_valid_per_rank : (rank + 1) * n_valid_per_rank
            ]

            if rank == 0:
                n_drop_train = len(all_train_shards) - n_train_per_rank * world_size
                n_drop_valid = len(all_valid_shards) - n_valid_per_rank * world_size
                # Estimate graphs/rank from manifest totals if available,
                # otherwise fall back to summing shard counts.
                n_train_total = manifest.get('n_train') or sum(
                    c for _, c in all_train_shards)
                n_valid_total = manifest.get('n_valid') or sum(
                    c for _, c in all_valid_shards)
                print(f"\033[32mStreaming sharded cache: {cache_dir}\033[0m")
                print(f"  Train: {n_train_per_rank} shards/rank "
                      f"({n_drop_train} shard(s) dropped for DDP balance) "
                      f"→ ~{n_train_per_rank * (n_train_total // len(all_train_shards)):,} graphs/rank")
                print(f"  Valid: {n_valid_per_rank} shards/rank "
                      f"({n_drop_valid} shard(s) dropped) "
                      f"→ ~{n_valid_per_rank * (n_valid_total // len(all_valid_shards)):,} graphs/rank")

            # ShardedDataset streams one shard at a time — peak RAM = 1 shard
            train_dataset = ShardedDataset(
                my_train_shards, cache_dir, shuffle=True, seed=random_seed
            )
            valid_dataset = ShardedDataset(
                my_valid_shards, cache_dir, shuffle=False, seed=random_seed
            )

            # [ablation hook] env overrides for RSS leak hunt
            _nw = int(os.environ.get("BAM_NUM_WORKERS", "2"))
            _pm = os.environ.get("BAM_PIN_MEMORY", "1") == "1"
            train_loader = DataLoader(
                train_dataset, batch_size=nbatch,
                num_workers=_nw,
                pin_memory=_pm,
                drop_last=True,
                persistent_workers=(_nw > 0),
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=nbatch,
                num_workers=0,
                pin_memory=_pm,
                drop_last=False,
            )
            return train_loader, valid_loader, uniq_element, enr_avg_per_element

    # --- Original path: read XYZ, convert, cache ---
    cache_path = _get_cache_path(
        fname, ntrain, nvalid, random_seed, cutoff,
        charge_key, regress_forces, max_neigh,
    )

    if rank == 0:
        if os.path.exists(cache_path):
            # Load from existing cache
            print(f"\033[32mLoading cached dataset: {cache_path}\033[0m")
            cached = torch.load(cache_path, weights_only=False)
            train_graphs = cached['train_graphs']
            valid_graphs = cached['valid_graphs']
            uniq_element = cached['uniq_element']
            enr_avg_per_element = cached['enr_avg_per_element']
        else:
            # Read and convert from scratch
            traj = read(fname, index=':')

            if element is None or element == "auto":
                element = sorted(
                    list(set(int(atom.number)
                             for atoms in traj for atom in atoms))
                )
            elif isinstance(element, str):
                element = [int(e) for e in element.split()]

            enr_avg_per_element, uniq_element, enr_var = \
                get_enr_avg_per_element(traj, element)

            # Stratified train/valid split by (total_charge, total_multiplicity)
            # Ensures each charge/spin group is proportionally represented
            rng = np.random.RandomState(random_seed)

            groups = {}
            for i, atoms in enumerate(traj):
                tc = atoms.info.get(total_charge_key, 0.0)
                tm = atoms.info.get('total_multiplicity', 1)
                gk = (float(tc), int(tm) if isinstance(tm, (int, float)) else 1)
                groups.setdefault(gk, []).append(i)

            if len(groups) > 1:
                # Stratified: allocate train/valid proportionally per group
                train_idx = []
                valid_idx = []
                for gk in sorted(groups):
                    g_idx = np.array(groups[gk])
                    rng.shuffle(g_idx)
                    frac = len(g_idx) / len(traj)
                    n_tr = max(1, int(round(ntrain * frac)))
                    n_va = max(1, int(round(nvalid * frac)))
                    n_tr = min(n_tr, len(g_idx))
                    n_va = min(n_va, len(g_idx) - n_tr)
                    train_idx.extend(g_idx[:n_tr])
                    valid_idx.extend(g_idx[n_tr:n_tr + n_va])
                    print(f"  \033[36mGroup charge={gk[0]:+.0f} mult={gk[1]}: "
                          f"{len(g_idx)} total -> "
                          f"{n_tr} train / {n_va} valid\033[0m")
                train_data = [traj[i] for i in train_idx]
                valid_data = [traj[i] for i in valid_idx]
            else:
                # Single group: fallback to plain shuffle
                idx = rng.permutation(len(traj))
                train_data = [traj[i] for i in idx[:ntrain]]
                valid_data = [traj[i] for i in idx[ntrain:ntrain + nvalid]]

            del traj, groups  # free ASE trajectory memory

            print(f"\n\033[32mConverting training data ({ntrain}) "
                  f"with charge info...\033[0m")
            train_graphs = get_graphset_charge(
                train_data, cutoff, uniq_element,
                enr_avg_per_element, enr_var,
                regress_forces=regress_forces,
                max_neigh=max_neigh,
                show_progress=True,
                desc="Train",
                charge_key=charge_key,
                total_charge_key=total_charge_key,
            )
            del train_data

            print(f"\033[32mConverting validation data ({nvalid}) "
                  f"with charge info...\033[0m")
            valid_graphs = get_graphset_charge(
                valid_data, cutoff, uniq_element,
                enr_avg_per_element, enr_var,
                regress_forces=regress_forces,
                max_neigh=max_neigh,
                show_progress=True,
                desc="Valid",
                charge_key=charge_key,
                total_charge_key=total_charge_key,
            )
            del valid_data

            # Save cache for other ranks (and future reruns)
            if world_size > 1:
                print(f"\033[32mSaving dataset cache: {cache_path}\033[0m")
                torch.save({
                    'train_graphs': train_graphs,
                    'valid_graphs': valid_graphs,
                    'uniq_element': uniq_element,
                    'enr_avg_per_element': enr_avg_per_element,
                }, cache_path)

    # File-based sync: avoids NCCL timeout during long conversions
    if world_size > 1:
        _wait_for_cache(cache_path, rank)

    # Non-rank-0: load from cache
    if rank != 0:
        cached = torch.load(cache_path, weights_only=False)
        train_graphs = cached['train_graphs']
        valid_graphs = cached['valid_graphs']
        uniq_element = cached['uniq_element']
        enr_avg_per_element = cached['enr_avg_per_element']
        del cached

    # Create DataLoader (DDP: partition data with DistributedSampler)
    train_sampler = None
    valid_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_graphs, num_replicas=world_size, rank=rank, shuffle=True
        )
        valid_sampler = DistributedSampler(
            valid_graphs, num_replicas=world_size, rank=rank, shuffle=False
        )

    train_loader = DataLoader(
        train_graphs, batch_size=nbatch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_graphs, batch_size=nbatch,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1),
        drop_last=False,
    )

    return train_loader, valid_loader, uniq_element, enr_avg_per_element


def get_dataloader_charge_to_predict(
    fname, ndata, nbatch, cutoff, model_ckpt,
    regress_forces=True, max_neigh=None,
    charge_key="charges", total_charge_key="total_charge",
    cache_dir=None,
):
    """
    Create evaluation DataLoader (with charge information).

    If cache_dir is provided and contains manifest_cd.pt, uses ShardedDataset
    to stream shard files one at a time — peak RAM = 1 shard, no accumulation.
    This is required when predicting on large datasets (e.g., full 2M QM9star).

    Args:
        fname: structure file path
        ndata: number of data (file path or integer)
        nbatch: batch size
        cutoff: distance cutoff
        model_ckpt: trained model checkpoint
        regress_forces: whether to regress forces
        max_neigh: maximum number of neighbors
        charge_key: atomic charge key name
        total_charge_key: total charge key name
        cache_dir: pre-built sharded cache directory (overrides XYZ conversion)

    Returns:
        data_loader, uniq_element, enr_avg_per_element
    """
    # Restore element information from checkpoint
    uniq_element = model_ckpt['uniq_element']
    enr_avg_per_element = model_ckpt['enr_avg_per_element']

    # --- Shard cache path: stream all shards without accumulation ---
    if cache_dir is not None:
        manifest_path = os.path.join(cache_dir, 'manifest_cd.pt')
        if os.path.exists(manifest_path):
            manifest = torch.load(manifest_path, weights_only=False)
            all_shards = manifest['train_shards'] + manifest['valid_shards']
            n_total = manifest.get('n_train', 0) + manifest.get('n_valid', 0)
            print(f"\033[32mPredict: streaming {n_total:,} graphs "
                  f"from {len(all_shards)} shards\033[0m")
            dataset = ShardedDataset(all_shards, cache_dir,
                                     shuffle=False, seed=0)
            data_loader = DataLoader(dataset, batch_size=nbatch,
                                     num_workers=2, pin_memory=True,
                                     drop_last=False)
            return data_loader, uniq_element, enr_avg_per_element

    # --- Original path: read XYZ, convert all at once (small datasets only) ---
    enr_var = 1.0

    # Read data
    if isinstance(ndata, str):
        traj = read(ndata, index=':')
    else:
        traj = read(fname, index=f':{ndata}')

    # Create graph dataset
    graphs = get_graphset_charge(
        traj, cutoff, uniq_element,
        enr_avg_per_element, enr_var,
        regress_forces=regress_forces,
        max_neigh=max_neigh,
        show_progress=True,
        desc="Predict",
        charge_key=charge_key,
        total_charge_key=total_charge_key,
    )

    data_loader = DataLoader(graphs, batch_size=nbatch, shuffle=False)
    return data_loader, uniq_element, enr_avg_per_element
