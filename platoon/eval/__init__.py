"""Evaluate a policy on environments: ``platoon eval`` (:mod:`platoon.eval.run`).

The config's ``inference:`` section (:class:`EvalConfig`) names the policy and where reports go; the section keeps
its name because training configs already use ``eval:`` for evaluation during training.
"""

from platoon.eval.config_defs import EvalConfig, EvalPolicy, EvalWorkflowConfig, HostedConfig, PricingConfig
from platoon.eval.runner import EvalRunner
from platoon.eval.workflow import (
    DefaultEvalGroupWorkflow,
    EvalRolloutRecord,
    EvalWorkflow,
)

__all__ = [
    "DefaultEvalGroupWorkflow",
    "EvalConfig",
    "EvalPolicy",
    "EvalRolloutRecord",
    "EvalRunner",
    "EvalWorkflow",
    "EvalWorkflowConfig",
    "HostedConfig",
    "PricingConfig",
]
