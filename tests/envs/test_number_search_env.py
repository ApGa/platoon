"""Number search, the example environment that ships with platoon (``platoon.envs.number_search``): its tasks come
from a seed, its environment imports only openreward and pydantic, its prompt builder is named in CodeAct's
options, and ``platoon-number-search@1`` resolves as a built-in, with nothing installed.

The SHA-256 values pin the tasks to the files number search read them from until it generated them, byte for byte.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

from platoon import extensions
from platoon.envs.named import BUILTIN_ENVIRONMENTS, expand_environment, get_environment
from platoon.envs.number_search import env
from platoon.envs.number_search.prompts import NumberSearchPromptBuilder
from platoon.harness.builtin.codeact import CodeActOptions

PACKAGE = Path(env.__file__).parent
# The task files' SHA-256 (one JSON line per task spec, in the order generated), which the generated tasks keep.
SHA256 = {
    "train": "cda011f13b3c0427222d75fecaabb178e45509911369bd8d51d366d3d0bd768f",
    "val": "62ce7ad185ce5adb0542007e63249c274dcd114e9ff6b21659f450680e1972b1",
}
SERVICE = "python://platoon.envs.number_search.env:NumberSearchEnvironment"
PROMPT_BUILDER = "platoon.envs.number_search.prompts:NumberSearchPromptBuilder"


def _imports(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    return imported


def test_the_generated_tasks_are_the_tasks_number_search_always_had():
    for split, digest in SHA256.items():
        rows = "".join(json.dumps(spec) + "\n" for spec in env.task_specs(split)).encode()
        assert hashlib.sha256(rows).hexdigest() == digest, split

    assert (len(env.task_specs("train")), len(env.task_specs("val"))) == (50_000, 1_000)
    assert env.task_specs("val")[0] == {
        "goal": "Guess the correct number between 119 and 965.",
        "id": "number_search.val.0",
        "max_steps": 1,
        "misc": {"low": 119, "high": 965, "target": 447},
    }
    assert env.NumberSearchEnvironment.list_tasks("val")[0] == env.task_specs("val")[0]
    with pytest.raises(ValueError, match="Unknown number-search split 'test'"):
        env.task_specs("test")


def test_the_environment_imports_only_openreward_pydantic_and_its_tasks():
    standard = set(sys.stdlib_module_names) | {"__future__"}
    environment = sorted(_imports(PACKAGE / "env.py") - {"platoon.envs.number_search.tasks"})
    tasks = sorted(_imports(PACKAGE / "tasks.py"))

    assert [name for name in environment if name.split(".")[0] not in standard | {"openreward", "pydantic"}] == []
    assert [name for name in tasks if name.split(".")[0] not in standard] == []  # the standard library only


def test_codeact_options_name_the_prompt_builder():
    builder = CodeActOptions.from_mapping({"prompt_builder": PROMPT_BUILDER, "include_reasoning": False})
    number_search = builder.make_prompt_builder()

    assert isinstance(number_search, NumberSearchPromptBuilder) and not number_search.include_reasoning
    assert number_search.describe_action_space([]) == ""  # the earlier number-search agent showed none


def test_platoon_number_search_is_a_built_in_that_needs_nothing_installed(monkeypatch):
    def no_entry_points(kind: str, name: str) -> list:
        raise AssertionError(f"looked up the entry points named {name!r} of {kind}")

    monkeypatch.setattr(extensions, "load_named", no_entry_points)

    entry = get_environment("platoon-number-search")

    assert entry.ref == "platoon-number-search@1" and get_environment("platoon-number-search@1") is entry
    origin = extensions.recorded_origin("environment", entry.ref)
    assert (origin.distribution, origin.version, origin.entry_point) == ("platoon", extensions.platoon_version(), None)
    expanded = expand_environment({"env": "platoon-number-search@1", "harness": "platoon-codeact"})
    assert (expanded["service"], expanded["harness_options"]["prompt_builder"]) == (SERVICE, PROMPT_BUILDER)
    [listed] = [info for info in extensions.list_components("environment") if info.name == entry.ref]
    assert (listed.builtin, listed.distribution, listed.value) == (
        True,
        "platoon",
        BUILTIN_ENVIRONMENTS["platoon-number-search"],
    )
