"""
CDEvaluatorV4 (BAM-torch CD Ver.2, 2026-05-21~).

Derived from CDEvaluatorV3 (Ver.1). Only `set_model` is overridden so the
ChargeRACEv4 model (with U_SLR) is instantiated; the holdout/test_values loop
is inherited unchanged from CDEvaluatorV3.
"""

from bam_torch.charge_dependent.predicting.cd_evaluator_v3 import CDEvaluatorV3
from bam_torch.charge_dependent.v2.model.cd_model_v4 import ChargeRACEv4
from bam_torch.utils.model_config import (
    parse_model_config,
    parse_cueq_config,
    parse_charge_config,
)


class CDEvaluatorV4(CDEvaluatorV3):
    """Phase 4 (Ver.2) evaluator using V3's evaluation path + V4 model."""

    def set_model(self):
        """Configure ChargeRACEv4 (Ver.2, CENT2 full = self + U_SLR) model.

        Mirrors CDTrainerV4.set_model() so trainer/evaluator produce identical
        model instances given the same input.json.
        """
        model_kwargs = parse_model_config(self.json_data)

        cueq_config = parse_cueq_config(self.json_data)
        model_kwargs['cueq_config'] = cueq_config
        if cueq_config is not None:
            self.msg += '\nequiv. lib.:\n\033[33m -- CuEquivariance\033[0m\n'
        else:
            self.msg += '\nequiv. lib.:\n\033[33m -- e3nn\033[0m\n'

        cd_kwargs = parse_charge_config(self.json_data)
        model_kwargs.update(cd_kwargs)

        charge_cfg = self.json_data.get('charge', {})
        model_kwargs['slr_kernel'] = charge_cfg.get('slr_kernel', 'cent2')
        model_kwargs['slr_lambda'] = charge_cfg.get('slr_lambda', 0.340)  # 0.18 a_0^-1
        model_kwargs['slr_n'] = charge_cfg.get('slr_n', 4)
        model_kwargs['slr_sigma'] = charge_cfg.get('slr_sigma', 1.0)
        model_kwargs['slr_cutoff'] = charge_cfg.get('slr_cutoff', None)

        model = ChargeRACEv4(**model_kwargs)

        energy_mode = (
            "E_SR + U_CENT (= U_CEP_SELF + U_SLR)" if cd_kwargs['use_cent_energy']
            else "E_SR only (U_SLR not in E_total)"
        )
        self.msg += (
            f"\n\033[33m -- Ver.2 (Phase 4) evaluator: CENT2 FULL\033[0m\n"
            f"\033[33m -- charge_type : {cd_kwargs.get('charge_type', 'npa')}\033[0m\n"
            f"\033[33m -- energy mode : {energy_mode}\033[0m\n"
            f"\033[33m -- slr_kernel  : {model_kwargs['slr_kernel']}"
            f"  (λ={model_kwargs['slr_lambda']}, n={model_kwargs['slr_n']}, σ={model_kwargs['slr_sigma']})\033[0m\n"
        )
        return model
