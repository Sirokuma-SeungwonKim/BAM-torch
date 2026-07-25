"""
BAM-torch CD Ver.2 — CENT2 FULL paradigm (2026-05-21~).

E_total = E_SR + U_CENT
U_CENT  = U_CEP_SELF + U_SLR    (Ver.2 새: pairwise long-range Coulomb)

Registry 키:
  - model     : 'charge_race_v4', 'cd_race_v4'
  - trainer   : 'cd_v4', 'charge_dependent_v4'
  - evaluator : 'cd_v4', 'charge_dependent_v4'

Note: registry 등록은 top-level `bam_torch.training` / `bam_torch.charge_dependent.predicting`
의 `__init__.py` 안의 try/except 자동 hook 으로 처리됨. 이 패키지 __init__ 은
module-level eager import 를 피해 circular import 를 방지한다 (`register_v2()` 는
lazy 호출용 helper).
"""


def register_v2(MODEL_REGISTRY, TRAINER_REGISTRY, EVALUATOR_REGISTRY):
    """Register Ver.2 model/trainer/evaluator in BAM-torch top-level registries.

    Lazy helper — top-level __init__.py 의 자동 hook 외에 수동 등록 필요할 때만 사용.
    """
    from bam_torch.charge_dependent.v2.model.cd_model_v4 import ChargeRACEv4
    from bam_torch.charge_dependent.v2.training.cd_trainer_v4 import CDTrainerV4
    from bam_torch.charge_dependent.v2.predicting.cd_evaluator_v4 import CDEvaluatorV4

    MODEL_REGISTRY['charge_race_v4'] = ChargeRACEv4
    MODEL_REGISTRY['cd_race_v4'] = ChargeRACEv4
    TRAINER_REGISTRY['cd_v4'] = CDTrainerV4
    TRAINER_REGISTRY['charge_dependent_v4'] = CDTrainerV4
    EVALUATOR_REGISTRY['cd_v4'] = CDEvaluatorV4
    EVALUATOR_REGISTRY['charge_dependent_v4'] = CDEvaluatorV4


__all__ = ['register_v2']
