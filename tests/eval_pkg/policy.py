"""A policy that samples nothing, for rollouts whose harness never calls the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _NoRenderer:
    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> Any:
        raise AssertionError("the inert policy samples nothing")

    def parse_response(self, response: Any) -> Any:
        raise AssertionError("the inert policy samples nothing")


@dataclass(frozen=True)
class InertPolicy:
    """Installs a gateway that serves no model: ``harness_rollout`` needs one in every process it runs in."""

    def install(self) -> None:
        from platoon.gateway import ModelGateway, set_default_gateway

        set_default_gateway(ModelGateway(sampler=None, renderer=_NoRenderer(), tokenizer=None, model_name="inert"))
