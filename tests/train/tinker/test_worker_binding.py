"""Publishing and following the sampling client, and the proxy split.

Two silent failures are pinned here.  ``Renderer.__reduce__`` drops
``renderer_kwargs`` (which are applied by ``setattr`` after construction), so
shipping a pickled renderer to a worker would silently restore class defaults
the day someone sets a non-default value -- hence the name+kwargs rebuild.  And
a worker that *pinned* its sampling client would be a no-op under a full
fine-tuning gateway and a hard mid-rollout failure under a LoRA gateway, so
workers follow a republished binding instead.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import stat
import threading
from types import SimpleNamespace

import pytest

import platoon.train.tinker  # noqa: F401
from platoon.train.tinker import proxy
from platoon.train.workers import binding
from platoon.train.workers.binding import FollowingSamplingClient, publish_binding, read_binding


class _StubRenderer:
    """Just enough surface for TinkerLLM's constructor introspection."""

    def __init__(self) -> None:
        self.preserve_thinking = True
        self.enable_thinking = True

    def build_generation_prompt(self, messages, tools=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, tokens, tools=None):  # pragma: no cover
        raise NotImplementedError

    def get_stop_sequences(self):  # pragma: no cover
        return []


class _StubClient:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.samples = 0

    def sample(self, **kwargs):
        self.samples += 1
        return f"{self.tag}-sample"

    async def sample_async(self, **kwargs):
        self.samples += 1
        return f"{self.tag}-async"


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the tokenizer/processor/renderer lookups; no model download."""

    renderer = _StubRenderer()
    monkeypatch.setattr(proxy, "get_tokenizer", lambda name: object())
    monkeypatch.setattr(
        proxy.AutoProcessor, "from_pretrained", classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(OSError()))
    )
    monkeypatch.setattr(proxy, "get_renderer", lambda *args, **kwargs: renderer)
    monkeypatch.setattr(proxy, "register_platoon_renderers", lambda: None)
    return renderer


def test_build_tinker_llm_applies_renderer_kwargs(stub_model):
    """The silent-drift guard: a non-default value must actually stick."""

    llm = proxy.build_tinker_llm(
        model_name="Qwen/Qwen3",
        renderer_name="qwen3_6_preserve_thinking",
        sampling_client=_StubClient("a"),
        renderer_kwargs={"preserve_thinking": False},
    )
    assert llm.renderer.preserve_thinking is False
    assert llm.renderer.enable_thinking is True

    with pytest.raises(ValueError, match="does not support"):
        proxy.build_tinker_llm(
            model_name="Qwen/Qwen3",
            renderer_name="r",
            sampling_client=_StubClient("a"),
            renderer_kwargs={"no_such_attribute": 1},
        )


def test_build_tinker_llm_carries_sampling_defaults(stub_model):
    llm = proxy.build_tinker_llm(
        model_name="Qwen/Qwen3",
        renderer_name="r",
        sampling_client=_StubClient("a"),
        context_window_length=65536,
        max_tokens=1234,
        temperature=0.5,
        top_k=7,
        top_p=0.8,
        seed=11,
    )
    assert (llm.max_tokens, llm.temperature, llm.top_k, llm.top_p, llm.seed) == (
        1234,
        0.5,
        7,
        0.8,
        11,
    )
    assert llm.context_window_length == 65536


def test_build_tinker_llm_touches_no_global_state(stub_model, monkeypatch):
    """A worker must not open a second gateway session or replace the process's default gateway."""

    from platoon.gateway import core

    def explode(*args, **kwargs):
        raise AssertionError("build_tinker_llm must not construct a ServiceClient")

    monkeypatch.setattr(proxy.tinker, "ServiceClient", explode)
    monkeypatch.setattr(core, "_default_gateway", None)
    proxy.build_tinker_llm(model_name="Qwen/Qwen3", renderer_name="r", sampling_client=_StubClient("a"))
    assert core._default_gateway is None


def test_register_tinker_llm_still_populates_model_info(stub_model, monkeypatch):
    created = {}

    class FakeServiceClient:
        def create_sampling_client(self, base_model, model_path=None):
            created["base_model"] = base_model
            created["model_path"] = model_path
            return _StubClient("registered")

    from platoon.gateway import core

    monkeypatch.setattr(core, "_default_gateway", None)
    monkeypatch.setattr(proxy.tinker, "ServiceClient", FakeServiceClient)
    info = proxy.register_tinker_llm(
        "Qwen/Qwen3",
        "qwen3_6_preserve_thinking",
        context_window_length=4096,
        renderer_kwargs={"preserve_thinking": False},
        model_path="tinker://run/sampler_weights/000010",
    )
    assert info.model_name == "Qwen/Qwen3"
    assert core.default_gateway() is info.llm.gateway
    assert created["model_path"] == "tinker://run/sampler_weights/000010"
    # The two new fields let a worker rebuild an equivalent renderer without
    # reaching into trainer config.
    assert info.renderer_name == "qwen3_6_preserve_thinking"
    assert info.renderer_kwargs == {"preserve_thinking": False}
    assert created["base_model"] == "Qwen/Qwen3"


def test_publish_binding_is_atomic_and_private(tmp_path):
    path = str(tmp_path / "model_binding.pkl")
    publish_binding(_StubClient("v1"), 1, path)

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    client, version = read_binding(path)
    assert client.tag == "v1" and version == 1
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))


def test_a_concurrent_reader_never_sees_a_partial_binding(tmp_path):
    path = str(tmp_path / "model_binding.pkl")
    publish_binding(_StubClient("v0"), 0, path)
    seen = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                client, version = read_binding(path)
            except FileNotFoundError:
                continue
            seen.append((client.tag, version))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for version in range(1, 60):
            publish_binding(_StubClient(f"v{version}"), version, path)
    finally:
        stop.set()
        thread.join(timeout=10)

    assert seen, "the reader never observed a binding"
    # Every observation is a complete, self-consistent publication.
    for tag, version in seen:
        assert tag == f"v{version}"


def test_following_client_reloads_within_the_poll_interval(tmp_path, monkeypatch):
    # On a clock the test moves: on a loaded runner the wall clock can pass the interval during the publish.
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(binding, "time", SimpleNamespace(monotonic=lambda: clock.now))
    path = str(tmp_path / "model_binding.pkl")
    publish_binding(_StubClient("first"), 1, path)
    follower = FollowingSamplingClient(path, b"", poll_seconds=0.05)

    assert follower.sample() == "first-sample"
    assert follower.followed_version == 1

    clock.now += 0.02
    publish_binding(_StubClient("second"), 2, path)
    # Inside the interval the old client is still in service...
    assert follower.sample() == "first-sample"
    clock.now += 0.1
    # ... and after it, the new one is picked up without a restart.
    assert follower.sample() == "second-sample"
    assert follower.followed_version == 2


def test_following_client_does_not_restat_on_every_call(tmp_path, monkeypatch):
    path = str(tmp_path / "model_binding.pkl")
    publish_binding(_StubClient("first"), 1, path)
    follower = FollowingSamplingClient(path, b"", poll_seconds=30.0)

    stats = {"count": 0}
    real_stat = os.stat

    def counting_stat(target, *args, **kwargs):
        if target == path:
            stats["count"] += 1
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)
    for _ in range(200):
        follower.sample()
    # A sample loop must not turn into a stat storm on shared scratch.
    assert stats["count"] == 0


def test_a_corrupt_binding_leaves_the_previous_client_in_service(tmp_path, caplog):
    path = str(tmp_path / "model_binding.pkl")
    publish_binding(_StubClient("good"), 1, path)
    follower = FollowingSamplingClient(path, b"", poll_seconds=0.0)
    assert follower.sample() == "good-sample"

    with open(path, "wb") as handle:
        handle.write(b"this is not a pickle")
    with caplog.at_level("WARNING"):
        assert follower.sample() == "good-sample"
    assert any("binding" in record.message for record in caplog.records)
    assert follower.followed_version == 1

    # Recovery is automatic on the next valid publication.
    publish_binding(_StubClient("recovered"), 2, path)
    assert follower.sample() == "recovered-sample"


def test_bootstrap_blob_is_used_before_any_binding_file_exists(tmp_path):
    path = str(tmp_path / "missing.pkl")
    follower = FollowingSamplingClient(
        path, pickle.dumps(_StubClient("bootstrap")), poll_seconds=0.0, bootstrap_version=3
    )
    assert follower.sample() == "bootstrap-sample"
    assert follower.reload_count == 0
    assert follower.oldest_sampled_version == 3

    publish_binding(_StubClient("published"), 5, path)
    assert follower.sample() == "published-sample"
    assert follower.reload_count == 1
    assert follower.oldest_sampled_version == 3


def test_no_bootstrap_and_no_binding_fails_loudly_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        FollowingSamplingClient(str(tmp_path / "nope.pkl"), b"", poll_seconds=0.0)


def test_following_client_delegates_unknown_attributes(tmp_path):
    path = str(tmp_path / "model_binding.pkl")
    client = _StubClient("v1")
    publish_binding(client, 1, path)
    follower = FollowingSamplingClient(path, b"", poll_seconds=5.0)
    assert follower.tag == "v1"


def test_setup_wait_does_not_age_data_and_later_samples_keep_oldest_version(tmp_path):
    path = str(tmp_path / "binding.pkl")
    publish_binding(_StubClient("setup"), 3, path)
    follower = FollowingSamplingClient(path, poll_seconds=0)
    assert follower.current().tag == "setup"
    assert follower.oldest_sampled_version is None
    # Several updates while the environment is waiting for a CPU slot.
    publish_binding(_StubClient("first-generation"), 9, path)
    assert asyncio.run(follower.sample_async()) == "first-generation-async"
    assert follower.oldest_sampled_version == 9
    # Root/child/verifier calls can follow newer weights, but earlier sampled
    # tokens remain in the training datum and must retain their older age.
    publish_binding(_StubClient("later-generation"), 12, path)
    assert follower.sample() == "later-generation-sample"
    assert follower.followed_version == 12
    assert follower.oldest_sampled_version == 9


def test_equal_size_and_timestamp_publications_still_reload(tmp_path):
    path = str(tmp_path / "binding.pkl")
    publish_binding(_StubClient("v1"), 1, path)
    first_stat = os.stat(path)
    follower = FollowingSamplingClient(path, poll_seconds=0)
    assert follower.sample() == "v1-sample"
    publish_binding(_StubClient("v2"), 2, path)
    os.utime(path, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))
    assert os.stat(path).st_size == first_stat.st_size
    assert follower.sample() == "v2-sample"
    assert follower.followed_version == 2
