"""Shared configuration definitions for Platoon.

This module contains configuration classes that are used across different
parts of the codebase (rollouts, training, inference, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from platoon.deprecations import removed_hint


@dataclass
class InferenceParams:
    """Inference parameters for rollout-time model calls (``RolloutConfig.inference_params``): every agent's default
    sampling, which a call's own parameters override unless the rollout's ``sampling_precedence`` is ``config``.

    Only the parameters a config sets apply.  Each defaults to None, which leaves it to the model gateway's own
    default (``ModelGateway``: temperature 1, top_p 1, no top-k cutoff and 4096 completion tokens), so a harness that
    sends no ``max_tokens`` gets the gateway's budget unless the config sets one; the example configs set theirs.
    """

    # Sampling temperature (None: the gateway's default, 1.0).
    temperature: float | None = None
    # Nucleus-sampling probability mass (None: the gateway's default, 1.0).
    top_p: float | None = None
    # Top-k cutoff: sample among the k most likely tokens, -1 for no cutoff (None: the gateway's default, -1).
    top_k: int | None = None
    # Completion-token limit of a call that sets none itself (None: the gateway's default, 4096).
    max_completion_tokens: int | None = None


@dataclass
class RolloutConfig:
    """How each rollout runs, in training, evaluation and ``platoon rollout``: its step and time limits, where it
    writes its files, and the default sampling of its model calls.
    """

    # Keys a release removed, with what replaces them (platoon.deprecations).
    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "propogate_root_success": removed_hint("0.3.0", "propagate_root_success"),
    }

    # The policy's model name (None: the trainer's or evaluator's model).
    model_name: str | None = None
    # Whether the rollout produces training data; the trainer and evaluator set it.
    train: bool = False
    # Step budget of the root agent, replacing the task's own when set (None: the task's, else unbounded). Only a task
    # table's rows carry a budget of their own (their ``max_steps``): an ORS task spec's ``max_steps`` is not read,
    # Harbor tasks have none, and a TOML harness spec then falls back to its own default.
    max_steps: int | None = None
    # Directory for the rollout's files, such as its event files under ``events/``. ``platoon train``, ``platoon
    # eval`` and ``platoon rollout`` set it inside their run directory whatever a config says, and warn when a config
    # sets it, so configs leave it out; a library caller building a RolloutConfig chooses it.
    output_dir: str = "rollout_results"
    # Log a line when each rollout finishes.
    verbose: bool = True
    # Time limit of the rollout's root agent, in seconds: its harness stops at the earliest of this, its TOML spec's
    # timeout_s and the task's own time limit for its agent (a Harbor task's [agent] timeout_sec times
    # agent_timeout_multiplier). None: no limit of its own. Deadlines on whole rollouts (a rollout worker's, an
    # evaluation slot's) count this, else that harness's own limit where the task's environment says what it needs
    # beyond it, such as starting sandboxes and grading, which they add.
    timeout: int | None = None
    # Per-step deadline in seconds, for harnesses that read one from the rollout config; core harnesses do not.
    step_timeout: int = 300
    # Return the trajectory collection as a dict rather than an object; the trainer and evaluator set it.
    return_dict: bool = False
    # Give every trajectory of a rollout the root's success (the ``single`` program adds the finalizer). Unset
    # means false.
    propagate_root_success: bool | None = None
    # Score every subagent 0 (the ``single`` program adds the finalizer, before root-success propagation).
    skip_subagent_reward_computation: bool = False
    # Master switch for verifier subagents and behavior judges on completed children, across the whole agent
    # tree; a child they would grade shares the episode's outcome instead. ``judges`` turns off every judge, these
    # included (an evaluation sets it with inference.judges).
    enable_subagent_verifiers: bool = True
    # Default sampling of every model call in the rollout; a call's own parameters override it, unless
    # ``sampling_precedence`` is ``config``.
    inference_params: InferenceParams = field(default_factory=InferenceParams)
    # Which wins when a policy call sets its own sampling parameters. ``request`` (the default): the call's own.
    # ``config``: a policy call samples with inference_params' temperature, top_p and top_k, never with its own (the
    # gateway's defaults stand in for those inference_params leaves unset); with inference_params'
    # max_completion_tokens when set, else its own; and with its own seed and its own stop sequences, by which its
    # harness parses the reply. ``aux`` and ``judge`` calls keep their own parameters either way, except the ``aux``
    # calls of a harness in a task sandbox, which sample as its policy calls.
    sampling_precedence: Literal["request", "config"] = "request"
    # Free-form settings for custom rollouts and harnesses.
    extra: dict[str, Any] = field(default_factory=dict)
    # Run the model calls that judge agents rather than solve the task: verifier agents and the behavior judges
    # beside them, parent-usage judges, and judge agents and calls. With false none run: a child its environment grades
    # by verifier shares the episode's outcome instead, and a judge agent or call is refused. ``platoon eval`` sets it
    # from ``inference.judges``; a training config may not turn it off.
    judges: bool = True

    def __post_init__(self) -> None:
        self.propagate_root_success = bool(self.propagate_root_success)
        # Support loading from plain dicts from config loaders and subprocess paths.
        if isinstance(self.inference_params, dict):
            self.inference_params = InferenceParams(**self.inference_params)
        if self.sampling_precedence not in ("request", "config"):
            raise ValueError(f"sampling_precedence must be request or config, got {self.sampling_precedence!r}")
