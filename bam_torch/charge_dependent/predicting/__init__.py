from .cd_evaluator import CDEvaluator
from .cd_evaluator_v3 import CDEvaluatorV3


EVALUATOR_REGISTRY = {
    "cd": CDEvaluator,
    "charge_dependent": CDEvaluator,
    "cd_v3": CDEvaluatorV3,
    "charge_dependent_v3": CDEvaluatorV3,
    "cd_e": CDEvaluatorV3,
}

try:
    from bam_torch.charge_dependent.v2.predicting.cd_evaluator_v4 import CDEvaluatorV4
    EVALUATOR_REGISTRY["cd_v4"] = CDEvaluatorV4
    EVALUATOR_REGISTRY["charge_dependent_v4"] = CDEvaluatorV4
except ImportError:
    pass
