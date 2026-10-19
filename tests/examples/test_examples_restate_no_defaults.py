"""The example configs set only what makes them what they are: none of their settings restates the default it would
get anyway, or the value a file it extends gives it, since readers copy the examples, and a restated default reads as
a setting that matters and stays behind when the default changes.

For each example config, each setting the file writes (each leaf of its YAML, a list of scalars counting as one) is
removed from a copy, and the config is loaded again: when it builds the same config as before, the setting restated
what it would get anyway.  Free-form mappings (``service_kwargs``, ``harness_options`` and the other ``*_kwargs``)
are compared as written, so their keys are never flagged: only the component behind them knows its defaults.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon.utils.config import ConfigError, resolve_config

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CONFIGS = sorted(EXAMPLES.glob("*/*.yaml"))

# (config, dotted setting) -> why the example states that default all the same. Keep it empty where you can: say it in
# a comment or in the example's README instead.
ALLOWED: dict[tuple[str, str], str] = {}


def _schema(data: dict[str, Any]) -> type:
    from platoon.eval.run import PlatoonEvalConfig
    from platoon.train.components import EnvironmentsConfig
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig

    if "inference" in data:
        return PlatoonEvalConfig
    return PlatoonTinkerRLTrainerConfig if "train" in data else EnvironmentsConfig


def _settings(data: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """The path of each setting a file writes: the leaves of its mappings, through lists of mappings."""

    if isinstance(data, dict):
        found = []
        for key, value in data.items():
            nested = isinstance(value, list) and value and all(isinstance(item, dict) for item in value)
            if path == () and key == "extends":
                continue
            if (isinstance(value, dict) and value) or nested:
                found += _settings(value, (*path, key))
            else:
                found.append((*path, key))
        return found
    return [setting for index, item in enumerate(data) for setting in _settings(item, (*path, index))]


def _without(data: dict[str, Any], setting: tuple[Any, ...]) -> dict[str, Any]:
    """``data`` without ``setting``, and without the mappings that leaves empty."""

    data = copy.deepcopy(data)
    parents = [data]
    for key in setting[:-1]:
        parents.append(parents[-1][key])
    del parents[-1][setting[-1]]
    for depth in range(len(setting) - 1, 0, -1):
        if parents[depth] == {}:
            del parents[depth - 1][setting[depth - 1]]
    return data


def _built(path: Path, schema: type) -> str:
    config = resolve_config(path, [], schema, warn=False).config
    data = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config
    return json.dumps(data, default=str, sort_keys=True)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.relative_to(EXAMPLES).as_posix())
def test_no_example_config_restates_a_default(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    example = tmp_path / config.parent.name
    shutil.copytree(config.parent, example, ignore=shutil.ignore_patterns("runs", "__pycache__", ".venv"))
    profile = tmp_path / "site.yaml"  # a site profile's runs, for an example that leaves log_path to it
    profile.write_text(f"runs: {tmp_path / 'runs'}\n")
    monkeypatch.setenv("PLATOON_SITE", str(profile))
    monkeypatch.chdir(example)
    path = example / config.name
    written = path.read_text()
    raw = yaml.safe_load(written)
    try:
        schema = _schema(resolve_config(path, [], None, warn=False).data)
        before = _built(path, schema)
    except ConfigError as exc:  # a named environment of a package that is not installed
        pytest.skip(f"{config.name} does not load here: {str(exc).splitlines()[-1]}")

    restated = []
    for setting in _settings(raw):
        path.write_text(yaml.safe_dump(_without(raw, setting), sort_keys=False))
        try:
            after = _built(path, schema)
        except ConfigError:
            continue  # a setting the config needs
        finally:
            path.write_text(written)
        dotted = ".".join(str(key) for key in setting)
        if after == before and (config.relative_to(EXAMPLES).as_posix(), dotted) not in ALLOWED:
            restated.append(dotted)

    assert restated == [], f"{config.relative_to(EXAMPLES)} restates what it gets anyway: {restated}"


def test_every_allowed_restatement_is_still_in_its_config() -> None:
    for (name, dotted), reason in ALLOWED.items():
        assert reason.strip(), f"say why {name} states {dotted}"
        data = yaml.safe_load((EXAMPLES / name).read_text())
        for key in dotted.split("."):
            data = data[int(key) if isinstance(data, list) else key]
