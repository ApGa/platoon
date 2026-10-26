"""One policy in the trainer process (:class:`Policy`): its spec, and its state as training moves it on.

Rollout workers never hold one: they sample the sampling client that the trainer publishes for them
(:mod:`platoon.train.policy.binding`).
"""

import tinker

from platoon.train.policy.spec import PolicyId, PolicySpec
from platoon.train.tinker.key_refusal import key_refusals
from platoon.train.tinker.proxy import ModelInfo


class Policy:
    """One policy: its spec, its model info (the renderer, the tokenizer and the model gateway's ``TinkerLLM``, which
    samples it and keeps its version), and what each start opens and moves on.  The training loops read and write the
    latter through their shared state, whose leader the policy is
    (:class:`~platoon.train.trainer.state.TrainLoopSharedState`)."""

    # Set as a start opens the policy: its Tinker service client (connect), the training client on that service, the
    # sampling client of its newest weights (the launch save's, then each step's) and its optimizer's parameters.
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient
    optimizer_params: tinker.AdamParams

    def __init__(self, spec: PolicySpec, info: ModelInfo) -> None:
        self.spec = spec
        self.info = info
        # The optimizer steps the policy took over the run, its earlier starts' among them.
        self.optimizer_steps = 0

    @property
    def id(self) -> PolicyId:
        return self.spec.id

    def connect(self) -> None:
        """Open the policy's Tinker service client, a session on its service (``spec.tinker_base_url``).  A key the
        service refuses (401, 403) ends training with one line that names the key and the service."""

        with key_refusals():
            self.service_client = tinker.ServiceClient(base_url=self.spec.tinker_base_url)
