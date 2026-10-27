"""A policy's published sampler at one version (:class:`PolicySnapshot`, which ``Policy.snapshot()`` takes), for
inference outside the rollouts: a trainer's own writers, teachers and judges, such as a step hook's side work.  Nothing
it samples is recorded or trained.

Imports no Tinker, so that ``platoon.experimental`` re-exports it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from platoon.train.policy.spec import PolicyId

if TYPE_CHECKING:
    from platoon.gateway.core import ModelGateway


class PolicyMoved(RuntimeError):
    """A snapshot refused a sample: its policy published a newer sampler before the call, or while it ran, so the
    sample would not be its version's."""


@dataclass(frozen=True)
class PolicySnapshot:
    """A policy's published sampler at ``version``, with what renders and parses its calls.  On a fully fine-tuned
    service a sampling client samples the newest weights whatever its path, so :meth:`sample` checks the version
    around each call; a trainer that samples during a step stops that work in its hook's ``drain()``, before the step
    publishes.  The check reads the version, not the weights: there, the step's own optimizer steps change what the
    sampler samples before the step publishes, so a sample is its version's only up to the step's first update."""

    policy: PolicyId
    version: int
    sampling_client: Any  # a tinker.SamplingClient
    renderer: Any
    tokenizer: Any
    model_name: str
    context_window_length: int | None = None
    # The bound on each sampling call through the gateway (None: TINKER_SAMPLE_TIMEOUT_SECONDS, else the default).
    sample_timeout_seconds: float | None = None
    # The policy's version now, read at each check (None: the snapshot never moves, as a test's may not).
    version_now: Callable[[], int] | None = field(default=None, repr=False, compare=False)

    def current(self) -> bool:
        """Whether the policy published nothing since: its version is still the snapshot's."""

        return self.version_now is None or self.version_now() == self.version

    async def sample(self, prompt: Any, params: Any, *, num_samples: int = 1) -> Any:
        """``num_samples`` samples of ``prompt`` (a ``tinker.ModelInput``) under ``params`` (``tinker.SamplingParams``)
        from the snapshot's sampler; ``PolicyMoved`` when the policy published before the call or by the time it
        returned, so that "this sample came from version v" holds."""

        self._refuse_moved("before the call")
        response = await self.sampling_client.sample_async(
            prompt=prompt, num_samples=num_samples, sampling_params=params
        )
        self._refuse_moved("while it ran")
        return response

    def gateway(self, *, context_window_length: int | None = None) -> "ModelGateway":
        """A model gateway (render, prepare, sample and parse calls) on the snapshot's sampler, with no recorder: its
        context window the policy's, or ``context_window_length`` when that is smaller."""

        from platoon.gateway.core import ModelGateway

        window = self.context_window_length
        if context_window_length is not None:
            window = context_window_length if window is None else min(window, context_window_length)
        return ModelGateway(
            sampler=self.sampling_client,
            renderer=self.renderer,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            context_window_length=window,
            sample_timeout_seconds=self.sample_timeout_seconds,
        )

    def _refuse_moved(self, when: str) -> None:
        if not self.current():
            raise PolicyMoved(
                f"policy {self.policy!r} published a new version {when}: this snapshot samples version {self.version}"
            )
