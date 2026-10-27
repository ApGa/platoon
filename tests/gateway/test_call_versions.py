"""Each model call records the policy version it sampled (RFC 0020, P5): the gateway's own version, which the
trainer's policy keeps, or the version a sampler that follows a republished binding pinned with the client it handed
out; none for a gateway that does not know it, and none for an answer that sampled nothing."""

from __future__ import annotations

import logging
import pickle
import time
from typing import Any

import pytest
from gateway_fakes import MODEL_NAME, Completion, sample_response
from test_gateway_core import USER, chat, open_root

from platoon.gateway.core import ChatRequest, ModelGateway


def test_a_call_records_the_version_its_gateway_serves_and_none_without_one(make_gateway, sampler):
    sampler.script = [Completion("one"), Completion("two"), Completion("three")]
    interactions: dict = {}
    gateway = make_gateway(version=3)
    agent = open_root(gateway, interactions)

    first = chat(gateway, agent, [USER])
    gateway.version = 4  # the trainer published a new policy
    second = chat(gateway, agent, [USER])
    assert (first.version, second.version) == (3, 4)
    assert [interactions[response.call_id].version for response in (first, second)] == [3, 4]

    unversioned = make_gateway()
    response = chat(unversioned, open_root(unversioned), [USER])
    assert response.version is None and response.interaction().version is None


def test_a_budget_stop_samples_nothing_and_records_no_version(make_gateway):
    gateway = make_gateway(version=7)
    assert gateway.budget_stop(open_root(gateway)).version is None


class _VersionedClient:
    """A picklable sampling client that answers with its weights version; with ``then``, it publishes that binding
    while it samples asynchronously, as a trainer that publishes while a call is in flight does."""

    def __init__(self, version: int, then: tuple[int, str] | None = None) -> None:
        self.version = version
        self.then = then

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        if self.then is not None:
            from platoon.train.workers.binding import publish_binding

            version, path = self.then
            publish_binding(_VersionedClient(version), version, path)
        return sample_response([1000 + self.version], stop_reason="length")

    def sample(self, *, prompt, sampling_params, num_samples):
        return sample_response([1000 + self.version], stop_reason="length")


def test_a_call_through_a_following_sampler_records_the_version_of_the_client_it_sampled(renderer, tokenizer, tmp_path):
    from platoon.train.workers.binding import FollowingSamplingClient, publish_binding

    path = str(tmp_path / "binding.pkl")
    publish_binding(_VersionedClient(3), 3, path)
    follower = FollowingSamplingClient(path, poll_seconds=0.0, bootstrap_version=3)
    # The gateway's own version is the one the worker's request was bound to; the follower's pinned one wins.
    gateway = ModelGateway(sampler=follower, renderer=renderer, tokenizer=tokenizer, model_name=MODEL_NAME, version=1)
    interactions: dict = {}
    agent = open_root(gateway, interactions)

    first = chat(gateway, agent, [USER])
    publish_binding(_VersionedClient(5), 5, path)
    second = chat(gateway, agent, [USER])
    assert (first.tokens, first.version) == ([1003], 3)
    assert (second.tokens, second.version) == ([1005], 5)
    assert [interactions[response.call_id].version for response in (first, second)] == [3, 5]
    assert follower.oldest_sampled_version == 3


def test_a_binding_published_while_a_call_samples_does_not_relabel_it(renderer, tokenizer, tmp_path):
    from platoon.train.workers.binding import FollowingSamplingClient, publish_binding

    path = str(tmp_path / "binding.pkl")
    publish_binding(_VersionedClient(6, then=(7, path)), 6, path)
    follower = FollowingSamplingClient(path, poll_seconds=0.0)
    gateway = ModelGateway(sampler=follower, renderer=renderer, tokenizer=tokenizer, model_name=MODEL_NAME)

    response = chat(gateway, open_root(gateway), [USER])
    assert (response.tokens, response.version) == ([1006], 6)  # sampled by version 6, which it pinned
    assert follower.followed_version == 6
    assert follower.pinned()[1] == 7  # the next call follows 7


class _Delegating:
    """A sampler that delegates every attribute it lacks, as a proxy does: its class has no ``pinned``."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_a_sampler_whose_class_has_no_pinned_samples_at_the_gateways_version(make_gateway, sampler):
    sampler.script = [Completion("one")]
    sampler.pinned = lambda: (None, 99)  # what a delegating proxy would answer for the name, read on an instance
    gateway = make_gateway(version=2)
    gateway.sampler = _Delegating(sampler)
    response = chat(gateway, open_root(gateway), [USER])
    assert response.version == 2 and response.content == "one"


def test_the_trainers_policy_keeps_its_version_in_its_gateway(renderer, tokenizer, sampler):
    from platoon.train.tinker.proxy import TinkerLLM

    llm = TinkerLLM(model_name=MODEL_NAME, renderer=renderer, tokenizer=tokenizer, sampling_client=sampler)
    assert llm.version == llm.gateway.version == 0
    llm.set_version(4)
    assert llm.gateway.version == 4
    assert llm.increment_version() == 5 and llm.gateway.version == 5
    llm.update_sampling_client(sampler)  # a publication: the sampler and its version together
    assert llm.version == llm.gateway.version == 6
    sampler.script = [Completion("one")]
    assert chat(llm.gateway, open_root(llm.gateway), [USER]).version == 6


def test_the_follower_pins_what_it_samples_and_counts_it_as_sampled(tmp_path):
    from platoon.train.workers.binding import FollowingSamplingClient, publish_binding

    path = str(tmp_path / "binding.pkl")
    publish_binding(_VersionedClient(8), 8, path)
    follower = FollowingSamplingClient(path, pickle.dumps(_VersionedClient(2)), poll_seconds=0.0, bootstrap_version=2)
    assert follower.oldest_sampled_version is None
    client, version = follower.pinned()
    assert (client.version, version) == (8, 8) and follower.oldest_sampled_version == 8


class _Racing(ModelGateway):
    """A gateway whose trainer publishes ``then`` (a sampler, then its version, as ``TinkerLLM`` writes them) right
    after the gateway's next read of the attribute ``after``, as a trainer on another thread can between two reads."""

    after: str | None = None
    then: tuple[Any, int] = (None, 0)

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name == object.__getattribute__(self, "after"):
            self.after = None
            self.sampler, self.version = object.__getattribute__(self, "then")
        return value


@pytest.mark.parametrize("after", ["sampler", "version"])
def test_a_publication_between_the_gateways_reads_never_labels_a_call_newer_than_what_sampled_it(
    renderer, tokenizer, after
):
    gateway = _Racing(
        sampler=_VersionedClient(10), renderer=renderer, tokenizer=tokenizer, model_name=MODEL_NAME, version=10
    )
    gateway.after, gateway.then = after, (_VersionedClient(11), 11)
    response = chat(gateway, open_root(gateway), [USER])
    sampled_by = response.tokens[0] - 1000
    assert gateway.version == 11  # the publication happened during the call
    assert response.version is not None and response.version <= sampled_by


class _SlowPin:
    """A sampler that follows a binding whose reload takes ``seconds``, or fails (``fails``)."""

    def __init__(self, inner: Any, seconds: float = 0.0, fails: bool = False) -> None:
        self.inner, self.seconds, self.fails = inner, seconds, fails

    def pinned(self) -> tuple[Any, int]:
        time.sleep(self.seconds)
        if self.fails:
            raise OSError("the binding could not be read")
        return self.inner, 5


def test_a_followed_bindings_reload_counts_as_sampling_in_its_time_and_its_failures(make_gateway, caplog):
    gateway = make_gateway()
    gateway.sampler = _SlowPin(_VersionedClient(5), seconds=0.2)
    response = chat(gateway, open_root(gateway), [USER])
    assert (response.tokens, response.version) == ([1005], 5)
    assert response.timing is not None and response.timing.sample_seconds >= 0.2
    blocking = gateway.sample_sync(gateway.prepare(ChatRequest.from_params([USER], {})))
    assert (blocking.tokens, blocking.version) == ([1005], 5)
    assert blocking.timing is not None and blocking.timing.sample_seconds >= 0.2

    gateway.sampler = _SlowPin(_VersionedClient(5), fails=True)
    with caplog.at_level(logging.ERROR), pytest.raises(OSError):
        chat(gateway, open_root(gateway), [USER])
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("sample_async failed")]
