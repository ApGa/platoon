"""A model gateway for rollouts that make no model call, as the reference harnesses (``platoon-oracle``,
``platoon-noop``) do: a call fails the test::

    collection = await run_harness_rollout(task, config, gateway=no_model_gateway(), runner=runner)
"""

from __future__ import annotations

from platoon.gateway import ModelGateway
from platoon.harness.rollout import EmbeddedGateway


class NoModel:
    """A renderer for a gateway that must never render: these harnesses make no model calls."""

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        raise AssertionError("no model call expected")

    def parse_response(self, tokens):
        raise AssertionError("no model call expected")

    def get_stop_sequences(self):
        return []


def no_model_gateway() -> EmbeddedGateway:
    return EmbeddedGateway(ModelGateway(sampler=None, renderer=NoModel(), tokenizer=None, model_name="none"))
