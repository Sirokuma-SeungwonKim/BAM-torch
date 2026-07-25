"""
Charge-dependent trainer Phase 2 for BAM-torch.

Training support for CENT2-based CEP model (ChargeRACE Phase 2).

Differences from Phase 1:
  - charge_mode parameter removed (CEP always active)
  - chg_cons_lambda removed (unnecessary with hard conservation)
  - cep_hidden_dim parameter added
  - compute_loss: only loss_q retained (loss_q_cons removed)
"""

import os
import json

import torch
import numpy as np
from e3nn import o3

from bam_torch.training.base_trainer import BaseTrainer
from bam_torch.model.wrapper_ops import CuEquivarianceConfig
from bam_torch.charge_dependent.model import MODEL_REGISTRY
from bam_torch.charge_dependent.utils.cd_utils import get_dataloader_charge
from bam_torch.charge_dependent.training.mtl import UncertaintyWeighter


class CDTrainer(BaseTrainer):
    """
    Phase 2 Charge-dependent model Trainer.

    Inherits BaseTrainer and overrides:
      - set_model()           : creates ChargeRACE (Phase 2, CEP) model
      - configure_dataloader(): DataLoader with charge information
      - compute_loss()        : energy/force loss + charge loss (CEP-based)

    Loss composition:
      total_loss = enr_lambda  * loss_E
                 + frc_lambda  * loss_F
                 + chg_lambda  * loss_Q   (q_pred vs NPA charges)
    """

    def __init__(self, json_data, rank=0, world_size=1):
        super().__init__(json_data, rank, world_size)

    def _mtl_method(self):
        """Multi-task loss-weighting method: 'none' (default) | 'uncertainty'."""
        return str(self.json_data.get('NN', {}).get('mtl_weighting', 'none')).lower()

    def configure_optimizer(self):
        """Build the base optimizer, then (opt-in) attach a learnable multi-task
        loss weighter as an extra param group.

        Enabled by ``NN.mtl_weighting == 'uncertainty'``. The default ('none') creates
        no weighter and leaves the optimizer untouched, reproducing the fixed-lambda
        behavior exactly. The weighter's log-variances are optimized with the base lr.

        NOTE: the weighter lives on the trainer (not the model) and is NOT tracked by
        EMA. It is also not part of the best-valid model checkpoint. It IS, however,
        persisted and restored by the resume system (``NN.resume_autosave``) via
        ``_collect_extra_resume_state`` / ``_restore_extra_resume_state`` below, so a
        12h-cut / hang resume continues with the learned log-variances rather than
        resetting them to unit weights. (A plain best-valid ``restart`` without the
        resume system still re-initializes the weighter to unit weights.)
        """
        self.mtl_weighter = None
        if self._mtl_method() == 'uncertainty':
            self.mtl_weighter = UncertaintyWeighter(("e", "f", "q")).to(self.device)

        if self.mtl_weighter is None:
            # No MTL weighter: identical to base (incl. base's restart opt_state load).
            return super().configure_optimizer()

        # MTL on: the weighter's log-vars form a SECOND optimizer param group. On a
        # restart the saved opt_state carries both groups, so the group must be added
        # BEFORE load_state_dict -- otherwise Adam's load_state_dict raises
        # "loaded state dict has a different number of parameter groups". Base
        # configure_optimizer loads opt_state immediately after creating the
        # single-group optimizer, so we replicate its create-then-restore flow here
        # with the extra group inserted in between.
        optimizer = self.set_optimizer()
        optimizer.add_param_group(
            {'params': list(self.mtl_weighter.parameters())}
        )
        if self.json_data['NN'].get('restart'):
            try:
                optimizer.load_state_dict(self.model_ckpt['opt_state'])
            except Exception as e:
                print(f"\033[33m[MTL] optimizer state not restored on restart: "
                      f"{e}\033[0m", flush=True)
        if getattr(self, 'rank', 0) == 0:
            print(f"[MTL] uncertainty weighting enabled "
                  f"(learnable log-vars over {self.mtl_weighter.task_keys})",
                  flush=True)
        return optimizer

    def _collect_extra_resume_state(self):
        """Persist the MTL Kendall weighter's learned log-variances in the resume
        checkpoint, so an uncertainty run resumes with its adapted task weights
        instead of resetting to unit. No-op under fixed-lambda weighting."""
        weighter = getattr(self, 'mtl_weighter', None)
        if weighter is None:
            return {}
        return {'mtl_state': weighter.state_dict()}

    def _restore_extra_resume_state(self, ckpt):
        """Restore the MTL Kendall weighter's log-variances from a resume checkpoint.

        The weighter's parameters are the same objects added to the optimizer as a
        param group, and ``load_state_dict`` copies values in-place, so the optimizer
        state restored just before this call stays consistent with them.
        """
        weighter = getattr(self, 'mtl_weighter', None)
        if weighter is None or not ckpt.get('mtl_state'):
            return
        try:
            weighter.load_state_dict(ckpt['mtl_state'])
            if getattr(self, 'rank', 0) == 0:
                lv = {k: round(v, 4) for k, v in weighter.log_vars().items()}
                print(f"\033[32m[RESUME][MTL] restored learned log-vars {lv}\033[0m",
                      flush=True)
        except Exception as e:
            print(f"\033[33m[RESUME][MTL] weighter state not restored: {e}\033[0m",
                  flush=True)

    def set_model(self):
        """Configure ChargeRACE Phase 2 (CEP) model."""
        mc = self.json_data

        cutoff            = mc.get('cutoff', 6.0)
        num_species       = mc.get('num_species', 4)
        avg_num_neighbors = mc.get('avg_num_neighbors', 30)
        hidden_irreps     = o3.Irreps(mc.get('hidden_channels', "64x0e+64x1o+64x2e"))
        features_dim      = mc.get('features_dim', 64)
        num_basis_func    = mc.get('num_radial_basis', 8)
        nlayers           = mc.get('nlayers', 3)
        max_ell           = mc.get('max_ell', 3)
        output_irreps     = mc.get('output_channels', "1x0e")
        active_fn         = mc.get('active_fn', "identity")

        regress_forces = mc.get('regress_forces', "auto")
        if regress_forces is True:
            regress_forces = "autograd"
        elif regress_forces is False:
            regress_forces = "false"

        # CEP config
        charge_config = mc.get('charge', {})
        cep_hidden_dim = charge_config.get('cep_hidden_dim', 64)

        # CuEquivariance config
        cueq_config = mc.get('cueq_config')
        if cueq_config is None or cueq_config:
            try:
                import cuequivariance as cue
                import cuequivariance_torch as cuet
                CUET_AVAILABLE = True
            except ImportError:
                CUET_AVAILABLE = False
            if CUET_AVAILABLE:
                cueq_config = CuEquivarianceConfig(
                    enabled=True,
                    layout="ir_mul",
                    group="O3_e3nn",
                    optimize_all=True,
                )
                self.msg += '\nequiv. lib.:\n\033[33m -- CuEquivariance\033[0m\n'
        else:
            cueq_config = None
            self.msg += '\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'

        model_name = mc["model"].lower()
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(
                f"Unknown charge-dependent model: {mc['model']}"
            )

        model = model_cls(
            cutoff=cutoff,
            avg_num_neighbors=avg_num_neighbors,
            num_species=num_species,
            max_ell=max_ell,
            num_basis_func=num_basis_func,
            hidden_irreps=hidden_irreps,
            nlayers=nlayers,
            features_dim=features_dim,
            output_irreps=output_irreps,
            active_fn=active_fn,
            regress_forces=regress_forces,
            cueq_config=cueq_config,
            cep_hidden_dim=cep_hidden_dim,
        )

        self.msg += '\n\033[33m -- Phase 2: CEP (CENT2-based), hard charge conservation\033[0m\n'
        return model

    def configure_dataloader(self):
        """Configure DataLoader with charge information."""
        jd = self.json_data
        charge_config = jd.get('charge', {})
        charge_key       = charge_config.get('charge_key', 'charges')
        total_charge_key = charge_config.get('total_charge_key', 'total_charge')

        train_loader, valid_loader, uniq_element, enr_avg_per_element = \
            get_dataloader_charge(
                jd['fname_traj'],
                jd['ntrain'],
                jd['nvalid'],
                jd['nbatch'],
                jd['cutoff'],
                jd['NN']['data_seed'],
                jd['element'],
                jd['regress_forces'],
                jd.get('max_neigh'),
                charge_key=charge_key,
                total_charge_key=total_charge_key,
                rank=self.rank,
                world_size=self.world_size,
                cache_dir=jd.get('cache_dir'),
            )
        return train_loader, valid_loader, uniq_element, enr_avg_per_element

    def compute_loss(self, preds, data):
        """
        Phase 2 loss:
          total = enr_lambda * loss_E
                + frc_lambda * loss_F
                + chg_lambda * loss_Q   (CEP q_i vs NPA charges)

        Charge conservation loss removed since CEP guarantees it via hard constraint.
        """
        # Base energy / force / stress loss
        loss = super().compute_loss(preds, data)

        charge_config = self.json_data.get('charge', {})
        q_lambda = self.json_data['NN'].get('chg_lambda', 1.0)

        # Atomic charge supervision (NPA charges etc.)
        if ("atomic_charges" in preds and "atomic_charges" in data):
            charge_target = data["atomic_charges"].flatten()
            charge_pred   = preds["atomic_charges"].flatten()

            loss_fn_key = charge_config.get("charge_loss", "mse")
            if loss_fn_key in self.loss_fn:
                loss_q = self.loss_fn[loss_fn_key](charge_pred, charge_target)
            else:
                loss_q = torch.nn.functional.mse_loss(
                    charge_pred, charge_target
                )

            loss["loss_q"] = loss_q
            loss["loss"]   = loss["loss"] + q_lambda * loss_q

        # Opt-in: replace the fixed-lambda total with learned uncertainty weighting.
        # (Per-task loss_e/loss_f/loss_q dict entries are left raw for logging.)
        weighter = getattr(self, "mtl_weighter", None)
        if weighter is not None:
            weighted = weighter({
                "e": loss.get("loss_e"),
                "f": loss.get("loss_f"),
                "q": loss.get("loss_q"),
            })
            # keep any non-task regularizer (e.g. L2) at its fixed weight
            l2_lambda = self.json_data.get("NN", {}).get("l2_lambda", 0)
            if l2_lambda and "loss_l2" in loss:
                weighted = weighted + l2_lambda * loss["loss_l2"]
            loss["loss"] = weighted

        # Optional dipole-MAGNITUDE supervision (opt-in NN.dip_lambda, default 0 =
        # off → no change). Added on top of the final E/F/Q total whether fixed-weight
        # or Kendall (dipole is NOT one of the Kendall tasks). Scheme-free observable:
        # |μ_pred| = |Σ q_i (r_i−centroid)| vs the QM9star DFT dipole MAGNITUDE. Only the
        # magnitude is used — the stored DFT dipole VECTOR is in a different orientation
        # frame than the coordinates (see cd_model_v4.forward), so a component-wise
        # (vector) loss would be invalid. μ_pred is e·Å; the DFT dipole is Debye.
        # CAVEATS: (1) a single scalar |μ| per molecule is a weak constraint on the
        # charge DISTRIBUTION — use it as an auxiliary signal ALONGSIDE the q loss
        # (chg_lambda>0), never alone. (2) For CHARGED molecules |μ| is
        # reference-dependent and the DFT gauge origin is unknown, so training on it is
        # only clean for NEUTRAL groups (G1/G2); on G3/G4 the target is inconsistent.
        dip_lambda = self.json_data['NN'].get('dip_lambda', 0.0)
        if (dip_lambda and preds.get("dipole") is not None
                and "dipole" in data):
            _EA2D = 4.80320   # e·Å -> Debye
            mu_pred_D = preds["dipole"].norm(dim=-1) * _EA2D            # (B,) Debye
            mu_tgt_D = data["dipole"].view(-1, 3).norm(dim=-1)         # (B,) Debye
            _has = data.get("has_dipole", None)
            _mask = (_has.view(-1) if _has is not None
                     else torch.ones_like(mu_tgt_D))
            _denom = _mask.sum().clamp(min=1.0)
            loss_mu = (((mu_pred_D - mu_tgt_D) ** 2) * _mask).sum() / _denom
            loss["loss_mu"] = loss_mu
            loss["loss"] = loss["loss"] + dip_lambda * loss_mu

        return loss

    def train_one_epoch(self, mode='train', data_loader=None):
        """CPU compatibility bypass + per-epoch MTL-weight logging.

        Logs the learned Kendall weights once per TRAIN epoch -- EVERY epoch, not
        just at ``log_interval`` -- because the weighter adapts them each epoch.
        ``initial_test`` uses ``mode='test'`` so it does not bump the counter.
        """
        if self.device == 'cpu' or str(self.device) == 'cpu':
            _orig_sync = torch.cuda.synchronize
            torch.cuda.synchronize = lambda *a, **kw: None
            try:
                out = super().train_one_epoch(mode, data_loader)
            finally:
                torch.cuda.synchronize = _orig_sync
        else:
            out = super().train_one_epoch(mode, data_loader)

        if mode == 'train':
            self._mtl_epoch = getattr(self, "_mtl_epoch", 0) + 1
            self._log_mtl_weights(self.start_epoch + self._mtl_epoch)
        return out

    def _log_mtl_weights(self, disp_epoch):
        """Append the current Kendall weights to ``<log-basename>_mtl_weights.csv``.

        Called once per TRAIN epoch (EVERY epoch), so the full per-epoch trajectory
        of the learned weights is saved. No-op unless ``NN.mtl_weighting ==
        'uncertainty'``. Only rank 0 writes (avoids duplicate DDP rows). The CSV row
        is written silently every epoch; a summary line is printed only every
        ``log_interval`` epochs to avoid log spam.

        CSV columns (per task ``e``/``f``/``q``):
          - ``w_*``      = effective loss multiplier ``0.5 * exp(-s)`` -- the weight
                           actually applied to each task loss, directly comparable to
                           the fixed ``enr_lambda / frc_lambda / chg_lambda``.
          - ``logvar_*`` = the raw learned log-variance ``s = log(sigma^2)``.
        """
        weighter = getattr(self, "mtl_weighter", None)
        if weighter is None or getattr(self, "rank", 0) != 0:
            return

        keys = list(weighter.task_keys)             # ("e", "f", "q")
        raw = weighter.weights()                    # exp(-s) per task
        logvar = weighter.log_vars()                # s = log(sigma^2) per task
        eff = {k: 0.5 * raw[k] for k in keys}       # 0.5*exp(-s) = loss multiplier

        log_name = self.json_data.get("train", {}).get("fname_log", "loss_train.out")
        csv_path = os.path.splitext(log_name)[0] + "_mtl_weights.csv"

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                cols = (["epoch"]
                        + [f"w_{k}" for k in keys]
                        + [f"logvar_{k}" for k in keys])
                f.write(",".join(cols) + "\n")
            row = ([str(disp_epoch)]
                   + [f"{eff[k]:.6g}" for k in keys]
                   + [f"{logvar[k]:.6g}" for k in keys])
            f.write(",".join(row) + "\n")

        interval = max(1, int(getattr(self, "log_interval", 1)))
        if disp_epoch % interval == 0:
            pretty = "  ".join(f"{k}={eff[k]:.4g}" for k in keys)
            print(f"[MTL] epoch {disp_epoch}: effective loss weights  {pretty}",
                  flush=True)

    def train(self):
        """Run base training, then persist the FINAL learned Kendall weights.

        Ensures the last-epoch weights are captured even if ``nepoch`` is not a
        multiple of ``log_interval`` (the per-epoch CSV may otherwise miss the final
        epoch). No behavioral change to training itself.
        """
        out = super().train()
        self._save_final_mtl_weights()
        return out

    def _save_final_mtl_weights(self):
        """Write the final learned weights to ``<log-basename>_final_mtl_weights.json``.

        Ready-to-reuse: to FREEZE these as fixed lambdas (e.g. take the weights a
        non-collapsing group converged to and apply them to a collapse-prone group),
        set ``NN.mtl_weighting = 'none'`` and copy ``suggested_fixed_lambda`` into
        ``NN.enr_lambda / NN.frc_lambda / NN.chg_lambda``. No-op under fixed-lambda
        weighting; only rank 0 writes.

        NOTE: the effective weight ``0.5*exp(-s)`` is the quantity directly
        comparable to a fixed lambda (it is the coefficient actually multiplying each
        task loss). Weights are per-dataset (loss scales differ per group), so
        transferring them across groups is a heuristic warm-start, not guaranteed
        optimal.
        """
        weighter = getattr(self, "mtl_weighter", None)
        if weighter is None or getattr(self, "rank", 0) != 0:
            return

        keys = list(weighter.task_keys)                 # ("e", "f", "q")
        raw = weighter.weights()                        # exp(-s) per task
        logvar = weighter.log_vars()                    # s per task
        eff = {k: 0.5 * raw[k] for k in keys}           # 0.5*exp(-s) = loss multiplier
        key2lambda = {"e": "enr_lambda", "f": "frc_lambda", "q": "chg_lambda"}

        log_name = self.json_data.get("train", {}).get("fname_log", "loss_train.out")
        json_path = os.path.splitext(log_name)[0] + "_final_mtl_weights.json"
        payload = {
            "note": ("Final learned Kendall weights. To FREEZE as fixed-lambda set "
                     "NN.mtl_weighting='none' and use 'suggested_fixed_lambda'."),
            "nepoch": self.json_data.get("NN", {}).get("nepoch"),
            "effective_weights": {k: eff[k] for k in keys},
            "suggested_fixed_lambda": {key2lambda.get(k, k): eff[k] for k in keys},
            "log_vars": {k: logvar[k] for k in keys},
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        pretty = "  ".join(f"{key2lambda.get(k, k)}={eff[k]:.6g}" for k in keys)
        print(f"\033[36m[MTL_FINAL] final learned weights -> {json_path}\n"
              f"           freeze as fixed-lambda:  {pretty}\033[0m", flush=True)
