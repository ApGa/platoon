"""Configuration for ``platoon eval``: the policy to sample and how to run the evaluation's rollouts.

The policy is a Tinker model (``model_name`` with its ``renderer_name``, as in training) or a hosted model
(``inference.hosted``): a base URL that speaks the dialect of each harness the run uses, behind the model gateway, for
evaluation and ``platoon rollout`` only (:mod:`platoon.gateway.hosted`).  A config never holds a key: ``api_key_env``
names the environment variable that does.

An evaluation writes to ``inference.output_dir``, or, when that is unset, to a directory of its own for each start,
``<runs>/eval/<policy>/<UTC time>``, where ``<runs>`` is the site profile's ``runs``, else ``runs`` under the working
directory (:data:`EVAL_RUNS_DIR`, :meth:`EvalConfig.policy_dir_name`).  It never resumes another policy's rollouts:
each rollout records the policy that made it (:func:`run_provenance`).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, ClassVar
from urllib.parse import urlsplit

from platoon.config_defs import RolloutConfig
from platoon.utils.config import ConfigError

# Where an evaluation that names no inference.output_dir writes: a new directory for each start,
# runs/eval/<policy>/<UTC time>, under the working directory, as platoon rollout writes to runs/adhoc/; under the site
# profile's runs instead when it has one (platoon.eval.run.prepare_output_dir).
EVAL_RUNS_DIR = os.path.join("runs", "eval")
# The longest name a policy's directory gets (EvalConfig.policy_dir_name): its end, where a checkpoint's name is.
_POLICY_DIR_NAME_MAX = 100

TINKER_API_KEY_VAR = "TINKER_API_KEY"
# What a process that serves an evaluation's Tinker policy sets before it opens its session (EvalPolicy.install).
# With telemetry on, the Tinker SDK reports every call that fails for good, such as a sample the server refused,
# with a SESSION_END event, and a self-hosted gateway such as miles-tinker reads that as the client's exit: it
# reclaims the session, and each later call of the process fails ("unknown sampling session"). An evaluation's
# worker process would then lose every rollout it runs after one failed call. Rollout workers turn it off in
# training for the same reason (platoon.train.workers.env).
TINKER_TELEMETRY_OFF = {"TINKER_TELEMETRY": "0"}
# The most of the Tinker service's answer that a refusal of the key repeats (TinkerKeyRefusedError).
_REFUSAL_DETAIL_MAX = 300


class TinkerKeyError(ConfigError):
    """``TINKER_API_KEY`` cannot open a session with the Tinker service: it is unset or lacks the ``tml-`` prefix the
    client requires (:func:`require_tinker_api_key`), or the service refused it (:class:`TinkerKeyRefusedError`).

    A configuration error (exit 78) of the environment rather than of a config file, so its message is the one line
    that says what is wrong, without the ``invalid config`` heading of the others.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.args = (message,)  # what str() gives, and what rebuilds the error when it is unpickled


class TinkerKeyRefusedError(TinkerKeyError):
    """The Tinker service refused ``TINKER_API_KEY``: HTTP 401 (a key it does not take), or 403 (a key it takes that
    may not do what was asked, such as a checkpoint that another key saved on a self-hosted service).

    Every request of the run would be refused alike, so ``platoon train``, ``eval`` and ``rollout`` end with its one
    line (exit 78) instead of the Tinker client's traceback, and an evaluation starts no rollout after the first that
    meets it (:meth:`platoon.eval.runner.EvalRunner.run_rollout_stage`).  ``status`` is the service's HTTP status.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def require_tinker_api_key() -> None:
    """Raise :class:`TinkerKeyError` where ``tinker.ServiceClient`` would refuse to start a session: with
    ``TINKER_API_KEY`` unset, or without the ``tml-`` prefix the pinned Tinker client (0.16.1) requires.

    ``platoon train``, ``eval`` and ``rollout`` call it (``eval`` and ``rollout`` when they serve a Tinker policy)
    before they prepare, write or reap anything, so a missing key is one line (exit 78) rather than the client's
    traceback, an evaluation whose every rollout errored, or a run directory and a reap of nothing.
    """

    key = os.environ.get(TINKER_API_KEY_VAR)
    if not key:
        raise TinkerKeyError(f"{TINKER_API_KEY_VAR} is not set: the policy is sampled through Tinker, which needs it")
    if not key.startswith("tml-"):
        raise TinkerKeyError(f"{TINKER_API_KEY_VAR} must start with tml-, as the Tinker client requires")


def tinker_key_refused(
    status: int, detail: str = "", *, service: str | None = None, asked: str | None = None
) -> TinkerKeyRefusedError:
    """The error for the Tinker service's refusal of ``TINKER_API_KEY`` with HTTP ``status``: ``detail`` is the
    service's answer, ``service`` its origin (scheme, host and port) and ``asked`` what it was asked for, unless it
    was a session.  The key itself is never in the message: an answer that repeats it has the variable's name in
    its place."""

    detail = " ".join(detail.split())
    key = os.environ.get(TINKER_API_KEY_VAR)
    if key:
        detail = detail.replace(key, TINKER_API_KEY_VAR)
    if len(detail) > _REFUSAL_DETAIL_MAX:
        detail = detail[: _REFUSAL_DETAIL_MAX - 3].rstrip() + "..."
    where = f"the Tinker service at {service}" if service else "the Tinker service"
    when = f" when asked for {asked}" if asked else ""
    answer = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
    return TinkerKeyRefusedError(
        f"{where} refused {TINKER_API_KEY_VAR}{when} ({answer}): set {TINKER_API_KEY_VAR} to a key that service "
        "accepts",
        status=status,
    )


def sample_timeout_note(key: str, seconds: float | None) -> str:
    """The line a run logs about its Tinker policy's sampling timeout: the bound on one sampling call and where it came
    from, the config's ``key`` (which holds ``seconds``; None: unset), else ``TINKER_SAMPLE_TIMEOUT_SECONDS``, else
    the default (:func:`platoon.gateway.core.resolve_sample_timeout`).  A key set beside the variable names the
    variable's value, which it wins over.

    ``platoon train`` and ``platoon eval`` log it before anything runs, so a variable that the key leaves the bound to
    and that the gateway would refuse raises :class:`ConfigError` (exit 78) there instead of failing each rollout.
    """

    from platoon.gateway.core import SAMPLE_TIMEOUT_ENV, resolve_sample_timeout

    try:
        resolved = resolve_sample_timeout(seconds)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    variable = os.environ.get(SAMPLE_TIMEOUT_ENV)
    if seconds is not None:
        source = key if variable is None else f"{key}, which wins over {SAMPLE_TIMEOUT_ENV}={variable}"
    elif variable is not None:
        source = f"{SAMPLE_TIMEOUT_ENV}; {key} is not set"
    else:
        source = f"the default; neither {key} nor {SAMPLE_TIMEOUT_ENV} is set"
    return f"Sampling timeout: {resolved:g} s ({source})"


@dataclass
class EvalWorkflowConfig:
    """Configuration for running evaluation rollouts over a dataset.

    A rollout counts only when its outcome was measured: one whose root outcome is not ``scored`` (a harness
    crash, a grader or env failure, an invalid grade, a timeout, a cancellation) is an errored rollout, like one
    that raised, and stays out of ``success_rate``, ``success_at_k`` and the reward statistics.
    ``count_timeout_as_failure`` counts a timed-out rollout as a valid failure instead (reward 0), as benchmarks
    that time-limit the policy do; off by default, since a timeout leaves the measurement incomplete, as in
    training.  ``fail_fast`` aborts the evaluation after the rollout stage when any rollout errored, unscored
    outcomes included.  With ``use_subprocesses``, rollouts run in up to ``subprocess_max_workers`` worker
    processes (default ``num_concurrent_workers``).

    ``success_threshold`` is the lowest reward that counts as a success.  A harness rollout is judged on the reward
    its environment scored, before any reward processor (an unscored one never succeeds); a legacy rollout
    function's own success flag, when it reports one, counts at 1.
    """

    # Rollouts of each task.
    num_rollouts_per_task: int = 1
    # Rollouts that run at once.
    num_concurrent_workers: int = 32
    # Lowest reward that counts as a success. A harness rollout counts by the reward its environment scored (an
    # unscored one never succeeds), a legacy rollout by its own success flag, when it reports one.
    success_threshold: float = 1.0
    # Stop after the rollout stage, without a report, when any rollout errored.
    fail_fast: bool = False
    # Count a timed-out rollout as a failure with reward 0 instead of an errored rollout.
    count_timeout_as_failure: bool = False
    # Run each rollout in a worker process, so a rollout that hangs or crashes costs only itself.
    use_subprocesses: bool = False
    # Worker processes with use_subprocesses (None: num_concurrent_workers).
    subprocess_max_workers: int | None = None
    # How each rollout runs: step and time limits, output directory, default sampling.
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class PricingConfig:
    """Prices in USD per million tokens, for the report's cost of a hosted model's calls."""

    # Prompt tokens that hit no cache.
    input: float
    # Completion tokens, reasoning tokens included.
    output: float
    # Prompt tokens read from the provider's prompt cache (None: the input price).
    cached_input: float | None = None
    # Prompt tokens written to the provider's prompt cache (None: the input price).
    cache_write: float | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"inference.hosted.pricing.{item.name} must be a non-negative number, got {value!r}")


# What replaced each key of the first draft's hosted backends (RFC 0015), for a config that still has one.
_FORWARDED_BODY = (
    "the gateway forwards the harness's own body: configure the harness or the proxy (LiteLLM's drop_params)"
)
# What a hosted model's api_key_env holds: the name of an environment variable, never a key.
_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STALE_HOSTED_KEYS = {
    "kind": "base_url speaks the dialect of the harness that calls it, so there is no kind",
    "require_api_key": "leave api_key_env out for a server without auth",
    "headers": "put headers in a proxy in front of the API",
    "headers_env": "put headers in a proxy in front of the API",
    "extra_body": _FORWARDED_BODY,
    "drop_params": _FORWARDED_BODY,
}


def _shown_url(url: Any) -> str:
    """``url`` for an error message: itself, unless it may hold a key (user info, a query or a fragment)."""

    if isinstance(url, str) and not any(mark in url for mark in "@?#"):
        return repr(url)
    return "a value with '@', '?' or '#', which this message does not repeat"


@dataclass
class HostedConfig:
    """``inference.hosted``: a hosted model to evaluate instead of a Tinker policy, behind the model gateway.

    Harnesses call the gateway in their own dialect, as in training, and the gateway forwards each call to the same
    path under ``base_url`` (``{base_url}/chat/completions``, ``/responses`` or ``/messages``), with the key from the
    variable ``api_key_env`` names.  So ``base_url`` must speak the dialect of every harness the run uses: an
    OpenAI-compatible server (OpenAI, vLLM, SGLang, OpenRouter) for Chat Completions, a Responses server for Codex, a
    Messages server for Claude Code, or a proxy that speaks several (a LiteLLM proxy).  Hosted calls are recorded
    without token ids, so they serve evaluation and ``platoon rollout`` only.
    """

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = _STALE_HOSTED_KEYS

    # The base URL, where the gateway's /v1 ends: https://api.openai.com/v1, https://api.anthropic.com/v1, a vLLM
    # server's http://localhost:8000/v1, a LiteLLM proxy's http://localhost:4000/v1. It must speak the dialect of every
    # harness the run uses.
    base_url: str
    # The environment variable that holds the API key (null: a server without auth, which gets no key). A config never
    # holds a key.
    api_key_env: str | None = None
    # Seconds one HTTP attempt may take, from sending the request to the reply's last byte.
    timeout_s: float = 600.0
    # Retries of 408, 409, 429 and 5xx replies, connection errors, timeouts, and streams cut off or failed midway, with
    # backoff that honours retry-after: a call makes max_retries + 1 attempts at most.
    max_retries: int = 6
    # Requests in flight at once for the whole run (null: no limit). Worker processes (use_subprocesses) split it, so it
    # must be at least their number.
    max_concurrent_requests: int | None = None
    # USD per million tokens, for the report's cost (null: the calls are left unpriced).
    pricing: PricingConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pricing, Mapping):
            self.pricing = PricingConfig(**self.pricing)
        # A message repeats neither a URL that may carry a key (in its user info, query or fragment) nor what
        # api_key_env holds, which may be a key put there by mistake: errors are printed and logged.
        parts = urlsplit(self.base_url) if isinstance(self.base_url, str) else None
        if parts is None or parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"inference.hosted.base_url must be an http or https URL, got {_shown_url(self.base_url)}")
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                "inference.hosted.base_url holds credentials, which a config never holds: put the key in the "
                "variable api_key_env names"
            )
        if parts.query or parts.fragment:
            raise ValueError(
                "inference.hosted.base_url must have no query or fragment, where a key would reach logs and lock "
                "files: put the key in the variable api_key_env names"
            )
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str) or not _VARIABLE_NAME.fullmatch(self.api_key_env)
        ):
            raise ValueError(
                "inference.hosted.api_key_env must be the name of an environment variable (letters, digits and "
                "underscores, such as OPENAI_API_KEY), or null: the key goes in that variable, never in a config"
            )
        if isinstance(self.timeout_s, bool) or not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError(f"inference.hosted.timeout_s must be a positive number, got {self.timeout_s!r}")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError(f"inference.hosted.max_retries must be a non-negative integer, got {self.max_retries!r}")
        limit = self.max_concurrent_requests
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError(f"inference.hosted.max_concurrent_requests must be a positive integer, got {limit!r}")


@dataclass(frozen=True)
class EvalPolicy:
    """What a rollout process needs to serve the policy: a Tinker model with its renderer and an optional
    checkpoint, or a hosted model (``hosted``).

    ``base_url`` is the Tinker server to sample from; None (an evaluation config, which has no such key) reads
    ``TINKER_BASE_URL``, else the hosted service.  ``platoon rollout`` of a training config sets it to the
    config's ``tinker_base_url``.  ``max_concurrent_requests`` is this process's share of a hosted model's request
    limit (:meth:`for_processes`).  ``sample_timeout_seconds`` bounds each sampling call of a Tinker policy; None
    leaves it to the variable ``TINKER_SAMPLE_TIMEOUT_SECONDS`` in the process that samples, else 600.
    """

    model_name: str
    renderer_name: str | None = None
    model_path: str | None = None
    renderer_kwargs: dict[str, Any] = field(default_factory=dict)
    context_window_length: int | None = None
    base_url: str | None = None
    hosted: HostedConfig | None = None
    max_concurrent_requests: int | None = None
    sample_timeout_seconds: float | None = None

    @property
    def is_hosted(self) -> bool:
        """Whether a hosted model serves the policy."""

        return self.hosted is not None

    def provenance(self) -> dict[str, Any]:
        """The policy's identity as each rollout's ``metadata.json`` records it: the backend and the model, and a
        Tinker policy's checkpoint and renderer or a hosted model's base URL origin."""

        if self.hosted is not None:
            from platoon.gateway.hosted import url_origin

            return {"backend": "hosted", "model": self.model_name, "base_url_origin": url_origin(self.hosted.base_url)}
        return {
            "backend": "tinker",
            "model": self.model_name,
            "model_path": self.model_path,
            "renderer": self.renderer_name,
        }

    def for_processes(self, processes: int) -> EvalPolicy:
        """This policy for one of ``processes`` worker processes: an equal share, rounded down, of a hosted model's
        ``max_concurrent_requests``, so that the run never has more requests in flight than the limit.

        Raises :class:`ConfigError` for a limit below ``processes``: each worker process makes one request at a time
        at least, so their requests in flight would exceed it.
        """

        limit = self.hosted.max_concurrent_requests if self.hosted is not None else None
        if limit is None or processes <= 1:
            return self
        if limit < processes:
            raise ConfigError(
                f"inference.hosted.max_concurrent_requests is {limit}, below the {processes} worker processes that "
                "share it (inference.workflow.subprocess_max_workers, at most num_concurrent_workers), each of which "
                f"makes a request at a time at least: raise it to {processes} or more, or run fewer worker processes"
            )
        return replace(self, max_concurrent_requests=limit // processes)

    def check_environment(self) -> None:
        """Raise :class:`ConfigError` for what this process lacks to serve the policy: a Tinker key
        (:func:`require_tinker_api_key`), or the variable a hosted model's key comes from.  Only the variable's name
        and whether it is set are read here."""

        if self.hosted is None:
            require_tinker_api_key()
            return
        name = self.hosted.api_key_env
        if name and not os.environ.get(name, "").strip():
            raise ConfigError(f"inference.hosted: set {name} in the environment, which api_key_env names")

    async def probe(self, dialects: Sequence[str]) -> dict[str, Any] | None:
        """Check a hosted model's base URL, key and model in each of ``dialects`` (the run's, :func:`run_dialects`)
        before any rollout: a :class:`ConfigError` that names the dialect, the base URL's origin and the provider's
        answer (exit 78) instead of a run of errored rollouts.  The model each dialect answered with, or None for a
        Tinker policy, which the first rollout checks."""

        if self.hosted is None:
            return None
        from platoon.gateway.hosted import HostedGateway, ProbeError, url_origin

        gateway = HostedGateway(self.hosted, self.model_name)
        where = url_origin(self.hosted.base_url)
        try:
            return await gateway.probe(dialects)
        except ProbeError as exc:
            verb = "refused" if exc.status is not None else "did not answer"
            raise ConfigError(f"inference.hosted: {where} {verb} the {exc.dialect} probe: {exc.detail}") from exc
        finally:
            await gateway.aclose()

    def lock_record(self, probe: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """``lock.json``'s ``policy``: the backend, the model and its base URL's origin; for a hosted model also the
        name of the variable its key comes from (never the key), its request limit and what the probe found."""

        from platoon.gateway.hosted import url_origin

        if self.hosted is None:
            return {
                "backend": "tinker",
                "model": self.model_name,
                "model_path": self.model_path,
                "renderer": self.renderer_name,
                "base_url_origin": url_origin(self.base_url or os.environ.get("TINKER_BASE_URL")),
            }
        return {
            "backend": "hosted",
            "model": self.model_name,
            "base_url_origin": url_origin(self.hosted.base_url),
            "api_key_env": self.hosted.api_key_env,
            "max_concurrent_requests": self.hosted.max_concurrent_requests,
            "probe": dict(probe) if probe is not None else None,
        }

    def install(self) -> None:
        """Serve the policy from this process: make its gateway the process's default, which rollouts serve.

        A Tinker policy samples through Tinker (raises :class:`ConfigError` without a usable ``TINKER_API_KEY``,
        :func:`require_tinker_api_key`, and :class:`TinkerKeyRefusedError` for a key the service refuses), with the
        Tinker SDK's telemetry off in this process (:data:`TINKER_TELEMETRY_OFF`); a hosted one forwards to its base
        URL (:class:`platoon.gateway.hosted.HostedGateway`), reading its key in this process at each call.
        """

        if self.hosted is not None:
            from platoon.gateway import set_default_gateway
            from platoon.gateway.hosted import HostedGateway

            gateway = HostedGateway(self.hosted, self.model_name, max_concurrent_requests=self.max_concurrent_requests)
            set_default_gateway(gateway)
            return

        from platoon.train.tinker.proxy import register_tinker_llm

        if self.renderer_name is None:
            raise ConfigError("A Tinker policy needs a renderer_name")
        require_tinker_api_key()
        os.environ.update(TINKER_TELEMETRY_OFF)  # before the session opens, which is when the SDK reads it
        register_tinker_llm(
            self.model_name,
            self.renderer_name,
            context_window_length=self.context_window_length,
            renderer_kwargs=dict(self.renderer_kwargs),
            model_path=self.model_path,
            base_url=self.base_url,
            sample_timeout_seconds=self.sample_timeout_seconds,
        )


@dataclass
class EvalConfig:
    """The ``inference:`` section of an evaluation config: the policy, the report directory and the rollouts.

    Harness rollouts reach the policy through a model gateway in each rollout
    process: ``model_name`` (or the Tinker sampler checkpoint ``model_path``),
    rendered with ``renderer_name``, or the hosted model ``hosted`` names.
    Without either no policy is served, which suits only custom rollouts that
    never call it: ``platoon eval`` then refuses environments that roll out
    with ``harness_rollout``, and the policy settings (``model_path``,
    ``renderer_kwargs``, ``sample_timeout_seconds``, ``context_window_length``)
    as well (:func:`unserved_policy_settings`), which ``platoon rollout``
    refuses only when no ``--renderer`` gives one.  A hosted model serves
    ``model_name`` itself, and refuses those Tinker settings.
    """

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "backend": "a hosted model is inference.hosted: {base_url, api_key_env}, where base_url speaks the harness's "
        "dialect (RFC 0015); kind, require_api_key, headers, headers_env, extra_body and drop_params are gone, and a "
        "Tinker server's URL is the TINKER_BASE_URL variable"
    }

    # The policy's model: with Tinker, the Hugging Face name of its base model; with a hosted model, the model id its
    # base URL serves (gpt-5.1, claude-opus-5-5, a vLLM server's served model name).
    model_name: str
    # Renderer of the policy's prompts with Tinker: a tinker-cookbook renderer name, or ``chat_template`` (None: no
    # Tinker policy is served).
    renderer_name: str | None = None
    # A Tinker sampler checkpoint (``tinker://...``) to sample instead of the base model.
    model_path: str | None = None
    # Keyword overrides for the renderer, set as its attributes. ``chat_template`` takes ``enable_thinking``,
    # ``preserve_thinking``, ``reasoning_effort``, ``chat_template_kwargs``, ``tool_call_parser`` (``auto``, a
    # parser's name, ``package.module:attr`` or ``none``) and ``reasoning_tags``; see
    # docs/how-to/models-and-tool-calling.md.
    renderer_kwargs: dict[str, Any] = field(default_factory=dict)
    # Seconds one sampling call of the Tinker policy may take before it fails (None: TINKER_SAMPLE_TIMEOUT_SECONDS, else
    # 600). The key wins over the variable, which a gateway job may export for every client it runs. A hosted model's
    # calls are bounded by hosted.timeout_s instead.
    sample_timeout_seconds: float | None = None
    # The model's context window in tokens; a prompt that does not fit is refused (None: no check).
    context_window_length: int | None = None
    # Directory of the rollouts, the report, ``config.resolved.yaml`` and ``lock.json`` (null: a new directory for each
    # start, ``runs/eval/<policy>/<UTC time>`` under the working directory, or under the site profile's ``runs`` when it
    # has one, where ``<policy>`` is the checkpoint, else the model). A directory that holds another policy's rollouts
    # is refused before anything in it changes.
    output_dir: str | None = None
    # How the evaluation's rollouts run.
    workflow: EvalWorkflowConfig = field(default_factory=EvalWorkflowConfig)
    # Reuse the rollouts already in output_dir instead of running them again; only ever this policy's, with the same
    # judges setting.
    resume: bool = True
    # With resume, run again the rollouts that errored.
    retry_errored: bool = True
    # Run the model calls that judge agents rather than solve the task: verifier agents and the behavior judges beside
    # them, parent-usage judges, and judge agents and calls. With false none run, which saves their latency and cost:
    # a child its environment grades by verifier shares the episode's outcome instead (an entry whose reward_processor
    # or program may read child verdicts is refused unless it sets agents.child_grading: none), and a judge agent or
    # call stops the evaluation (exit 78). Success metrics never depend on judges.
    judges: bool = True
    # A hosted model to evaluate instead of a Tinker policy: a base URL that speaks each harness's dialect, reached
    # through the model gateway, for evaluation only (null: Tinker). See docs/how-to/evaluate-hosted-models.md.
    hosted: HostedConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.hosted, Mapping):
            self.hosted = HostedConfig(**self.hosted)
        seconds = self.sample_timeout_seconds
        if seconds is not None and (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(
                f"inference.sample_timeout_seconds must be a positive number of seconds or null, got {seconds!r}"
            )
        if self.hosted is not None:
            tinker_only = [
                name
                for name, is_set in (
                    ("renderer_name", self.renderer_name is not None),
                    ("model_path", self.model_path is not None),
                    ("renderer_kwargs", bool(self.renderer_kwargs)),
                    ("sample_timeout_seconds", self.sample_timeout_seconds is not None),
                    ("context_window_length", self.context_window_length is not None),
                )
                if is_set
            ]
            if tinker_only:
                message = (
                    f"inference.{', inference.'.join(tinker_only)} configure a Tinker policy, but inference.hosted is "
                    "set, whose server renders its own prompts and serves inference.model_name; leave them out"
                )
                if "sample_timeout_seconds" in tinker_only:
                    message += " (inference.hosted.timeout_s bounds a hosted model's calls)"
                raise ValueError(message)

    def policy(self) -> EvalPolicy | None:
        """The policy rollouts sample: the hosted model, a Tinker model with its ``renderer_name``, or None (no
        renderer and no hosted model: nothing is served)."""

        if self.hosted is not None:
            return EvalPolicy(model_name=self.model_name, hosted=self.hosted)
        if self.renderer_name is None:
            return None
        settings: dict[str, Any] = {}
        if self.sample_timeout_seconds is not None:  # unset, the policy's own default stands
            settings["sample_timeout_seconds"] = self.sample_timeout_seconds
        return EvalPolicy(
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            model_path=self.model_path,
            renderer_kwargs=dict(self.renderer_kwargs),
            context_window_length=self.context_window_length,
            **settings,
        )

    def provenance(self) -> dict[str, Any]:
        """What each rollout of this evaluation records about it in its ``metadata.json`` (:func:`run_provenance`)."""

        return run_provenance(self.policy(), judges=self.judges)

    def policy_dir_name(self) -> str:
        """The policy as the name of one directory, the ``<policy>`` of the default output directory: the checkpoint
        ``model_path`` names, without its ``tinker://`` scheme, else ``model_name``, each run of characters other than
        letters, digits, ``.``, ``_`` and ``-`` made one ``-``, and the end kept of a name over 100 characters:
        ``Qwen/Qwen3-4B-Instruct-2507`` is ``Qwen-Qwen3-4B-Instruct-2507``, ``tinker://run-1/sampler_weights/000040``
        is ``run-1-sampler_weights-000040``."""

        name = (self.model_path or self.model_name or "").split("://", 1)[-1]
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
        return name[-_POLICY_DIR_NAME_MAX:].lstrip("-.") or "policy"


def run_provenance(policy: Any, *, judges: bool) -> dict[str, Any]:
    """What each rollout's ``metadata.json`` records about the evaluation that made it (``provenance``): whether its
    judges ran, and its policy's identity (:meth:`EvalPolicy.provenance`: the backend, the model, a Tinker policy's
    checkpoint and renderer, a hosted model's base URL origin), or ``backend: None`` when it serves no policy.  A
    policy object without ``provenance`` (a test's stand-in) adds nothing.

    An evaluation refuses an output directory whose rollouts record another
    (:func:`platoon.eval.runner.check_provenance`), so it never resumes another policy's rollouts."""

    record: dict[str, Any] = {"judges": bool(judges)}
    if policy is None:
        record["backend"] = None
    elif callable(getattr(policy, "provenance", None)):
        record.update(policy.provenance())
    return record


def unserved_policy_settings(
    renderer_name: str | None,
    *,
    model_path: str | None = None,
    renderer_kwargs: Mapping[str, Any] | None = None,
    context_window_length: int | None = None,
    sample_timeout_seconds: float | None = None,
) -> list[str]:
    """The policy settings given although no renderer serves the policy, which would drop them; none with a
    ``renderer_name``.  Checked once the renderer is final, since ``platoon rollout`` also takes it from
    ``--renderer``, never when a config loads."""

    if renderer_name is not None:
        return []
    given = (
        ("model_path", model_path is not None),
        ("renderer_kwargs", bool(renderer_kwargs)),
        ("sample_timeout_seconds", sample_timeout_seconds is not None),
        ("context_window_length", context_window_length is not None),
    )
    return [name for name, is_set in given if is_set]
