"""The renderers reproduce the saved token ids of every setting in use (``tokens.json.gz``, see ``renderer_cases``).

Each renderer is built by name with its ``renderer_kwargs`` and renders through the real ``ModelGateway``, so the
gateway's routing of tools and per-request ``chat_template_kwargs`` is covered too.  The simulated episode checks
that prefix merging (``trajectory_to_data``) cuts the same datums.
"""

from __future__ import annotations

import json
import os

import pytest
import renderer_cases as cases

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

pytestmark = pytest.mark.skipif(
    not all((path / "tokenizer.json").exists() for path in cases.SNAPSHOTS.values()),
    reason=f"Qwen3.6/3.8 preserve-thinking snapshots not available under {cases.HUB}",
)

SETTINGS = {setting: (snapshot, name, kwargs) for setting, snapshot, name, kwargs in cases.SETTINGS}


@pytest.fixture(scope="module")
def fixture():
    return cases.load_fixture()


@pytest.fixture(scope="module")
def tokenizers():
    return {key: cases.load_tokenizer(key) for key in cases.SNAPSHOTS}


def _gateway(tokenizers, setting):
    snapshot, name, kwargs = SETTINGS[setting]
    tokenizer = tokenizers[snapshot]
    return cases.make_gateway(cases.build_renderer(tokenizer, name, kwargs), tokenizer), tokenizer


def _first_difference(tokenizer, got: list[int], want: list[int]) -> str:
    index = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    window = slice(max(0, index - 8), index + 8)
    return (
        f"first difference at token {index} (lengths {len(got)} vs {len(want)}): "
        f"got {tokenizer.decode(got[window])!r}, saved {tokenizer.decode(want[window])!r}"
    )


@pytest.mark.parametrize("setting", list(SETTINGS))
def test_saved_prompt_tokens(fixture, tokenizers, setting):
    gateway, tokenizer = _gateway(tokenizers, setting)
    saved = fixture["settings"][setting]
    seen: set[str] = set()
    failures: list[str] = []
    for name, (tools, messages) in cases.all_conversations().items():
        for overrides in cases.OVERRIDES:
            key = f"{name} | {json.dumps(overrides, sort_keys=True)}"
            seen.add(key)
            want = saved[key]
            if isinstance(want, dict):
                with pytest.raises(Exception):
                    cases.render_ids(gateway, tools, messages, overrides)
                continue
            got = cases.render_ids(gateway, tools, messages, overrides)
            if got != want:
                failures.append(f"{key}: {_first_difference(tokenizer, got, want)}")
    assert not failures, "\n".join(failures)
    assert seen == set(saved)


@pytest.mark.parametrize("setting", [setting for setting, *_ in cases.SETTINGS if "disable thinking" not in setting])
def test_episode_datum_boundaries(fixture, tokenizers, setting):
    gateway, tokenizer = _gateway(tokenizers, setting)
    assert [list(datum) for datum in cases.run_episode(gateway, tokenizer)] == fixture["episodes"][setting]


def test_episode_breaks_only_where_re_rendering_differs(fixture):
    """Preserve-thinking renders cut one extra datum (the empty think block); preserve off also cuts at the nudge."""
    lengths = {setting: len(datums) for setting, datums in fixture["episodes"].items()}
    assert lengths == {
        "3.6 preserve": 2,
        "3.6 preserve off": 3,
        "3.8 xhigh (default)": 2,
        "3.8 xhigh": 2,
        "3.8 medium": 2,
        "3.8 low": 2,
    }
