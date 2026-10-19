"""CONTRIBUTING.md stays true: it shows the commands CI runs and the rules its hooks enforce, its code snippets
run, the outputs it shows are what platoon prints, and the code it points at exists.

The snippets are found by the comment on their first line (``# my_envs/repeat.py``) or by their first line of
output, so editing a snippet in the guide changes what these tests run.
"""

from __future__ import annotations

import asyncio
import importlib
import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from platoon import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "CONTRIBUTING.md"
TEXT = GUIDE.read_text()
# The GitHub Action that checks pull request titles, and its default types: the Conventional Commits types of the
# conventional-commit-types package.
TITLE_ACTION = "amannn/action-semantic-pull-request"
DEFAULT_PR_TITLE_TYPES = ("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert")
# The conventional-pre-commit hook's default types (DEFAULT_TYPES in its conventional_pre_commit/format.py).
DEFAULT_COMMIT_TYPES = ("build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test")

_FENCE = re.compile(r"^(?P<indent> *)```(?P<lang>[\w-]*)\s*$")


def code_blocks(text: str = TEXT) -> list[tuple[str, str]]:
    """``(language, code)`` of every fenced block, with the indentation of a block inside a list removed."""

    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = _FENCE.match(lines[index])
        index += 1
        if match is None:
            continue
        indent, body = match["indent"], []
        while lines[index].strip() != "```":
            line = lines[index]
            body.append(line[len(indent) :] if line.startswith(indent) else line.lstrip())
            index += 1
        index += 1
        blocks.append((match["lang"], "\n".join(body) + "\n"))
    return blocks


def block(first_line: str) -> str:
    """The one code block whose first line is ``first_line``."""

    found = [code for _, code in code_blocks() if code.splitlines()[0] == first_line]
    assert len(found) == 1, f"CONTRIBUTING.md has {len(found)} code blocks starting with {first_line!r}"
    return found[0]


def prose(text: str = TEXT) -> str:
    """The text outside fenced code blocks."""

    return re.sub(r"^ *```[\w-]*\s*\n.*?^ *```\s*$", "", text, flags=re.M | re.S)


def table(header: str) -> list[list[str]]:
    """The body rows of the guide's table whose header row starts with ``header``, as lists of cells."""

    lines = TEXT.splitlines()
    [start] = [index for index, line in enumerate(lines) if line.startswith(header)]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def workflows() -> dict[str, dict]:
    """Every GitHub Actions workflow in ``.github/workflows``, by file name."""

    return {
        path.name: yaml.safe_load(path.read_text()) for path in sorted((REPO_ROOT / ".github/workflows").glob("*.yml"))
    }


def workflow_steps(workflow: dict | None = None) -> list[dict]:
    """Every step of every job of ``workflow``, or of every workflow."""

    chosen = [workflow] if workflow is not None else list(workflows().values())
    return [step for each in chosen for job in each["jobs"].values() for step in job["steps"]]


def is_title_check(step: dict) -> bool:
    return str(step.get("uses", "")).startswith(TITLE_ACTION)


def title_check() -> dict:
    """The one workflow step that checks pull request titles."""

    checks = [step for step in workflow_steps() if is_title_check(step)]
    assert len(checks) == 1, f"CONTRIBUTING.md describes one pull request title check; the workflows have {len(checks)}"
    return checks[0]


def pre_commit_hooks() -> list[dict]:
    """Every hook in ``.pre-commit-config.yaml``, with its repository's ``rev``."""

    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    return [{**hook, "rev": repo.get("rev")} for repo in config["repos"] for hook in repo["hooks"]]


# --------------------------------------------------------------------------- CI and hooks


def test_the_commands_ci_runs_are_documented():
    commands = [step["run"].strip() for step in workflow_steps() if "run" in step]
    assert commands
    missing = [command for command in commands if command not in TEXT]
    assert not missing, f"CONTRIBUTING.md does not show these CI commands: {missing}"


def test_the_pull_request_title_rule_is_documented():
    check = title_check()
    configured = str((check.get("with") or {}).get("types", "")).split()
    assert check["uses"] in TEXT
    assert "Conventional Commits" in TEXT
    for kind in configured or DEFAULT_PR_TITLE_TYPES:
        assert f"`{kind}`" in TEXT, f"the PR title type {kind!r} is not listed"


def test_only_the_title_check_runs_again_when_a_pull_request_is_edited():
    """The guide says the title check runs again when the title changes; nothing else needs to (an edit of the
    description, a ticked checkbox) and skipped jobs would report success, so lint and tests do not listen."""

    for name, workflow in workflows().items():
        triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare ``on`` key as true
        pull_request = triggers.get("pull_request") if isinstance(triggers, dict) else None
        types = (pull_request or {}).get("types", [])
        steps = workflow_steps(workflow)
        if any(is_title_check(step) for step in steps):
            assert "edited" in types, f"{name} does not check a pull request's title again when it is edited"
        if "edited" in types:
            assert all(is_title_check(step) for step in steps), f"{name} runs more than the title check on every edit"


def test_the_pre_commit_hooks_are_documented():
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    for repo in config["repos"]:
        if repo.get("rev"):
            assert f"`{repo['rev']}`" in TEXT, f"the pinned version {repo['rev']} of {repo['repo']} is not documented"
        for hook in repo["hooks"]:
            assert f"`{hook['id']}`" in TEXT, f"the {hook['id']} hook is not documented"
            if "entry" in hook:
                assert f"`{hook['entry']}`" in TEXT, f"the {hook['id']} hook's command is not documented"


def test_the_ruff_hook_runs_the_ruff_that_uv_lock_installs():
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    [locked] = [package["version"] for package in lock["package"] if package["name"] == "ruff"]
    [rev] = {hook["rev"] for hook in pre_commit_hooks() if hook["id"] in ("ruff", "ruff-check", "ruff-format")}
    assert rev == f"v{locked}", f"the ruff hook runs {rev}, but uv.lock installs ruff {locked}"


def test_ty_runs_at_one_pinned_version():
    [hook] = [hook for hook in pre_commit_hooks() if hook["id"] == "ty"]
    assert re.fullmatch(r"uvx ty@\d+\.\d+\.\d+ check", hook["entry"]), f"ty is not pinned: {hook['entry']!r}"
    commands = set(re.findall(r"uvx ty(?:@\S+)? check", TEXT))
    assert commands == {hook["entry"]}, f"the guide runs ty as {sorted(commands)}, the hook as {hook['entry']!r}"


def test_ci_s_lint_job_runs_the_ruff_hooks():
    """``pre-commit run --all-files`` runs the hooks of the ``pre-commit`` stage; the ruff hooks must be among them."""

    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    default_stages = config.get("default_stages", ["pre-commit"])
    ruff = [hook for hook in pre_commit_hooks() if hook["id"] in ("ruff-check", "ruff-format")]
    assert len(ruff) == 2 and all("pre-commit" in hook.get("stages", default_stages) for hook in ruff)


def test_the_setup_installs_every_git_hook_the_config_uses():
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    [install] = re.findall(r"pre-commit install((?: --hook-type [\w-]+)+)", TEXT)
    installed = set(install.split()) - {"--hook-type"}
    default_stages = config.get("default_stages", ["pre-commit"])
    stages = {stage for hook in pre_commit_hooks() for stage in hook.get("stages", default_stages)}
    git_hooks = stages - {"manual"}
    assert git_hooks <= installed, f"the setup does not install the {sorted(git_hooks - installed)} hooks"
    # So that a plain `pre-commit install` installs them too.
    assert git_hooks <= set(config.get("default_install_hook_types", ["pre-commit"]))
    for hook in pre_commit_hooks():
        if "manual" in hook.get("stages", []):
            assert f"pre-commit run {hook['id']} --hook-stage manual" in TEXT, f"how to run {hook['id']} is not shown"


def test_the_commit_types_are_the_ones_the_hook_and_the_pr_title_check_accept():
    types = sorted(row[0].strip("`") for row in table("| Type | For |"))
    [hook] = [hook for hook in pre_commit_hooks() if hook["id"] == "conventional-pre-commit"]
    assert hook.get("stages") == ["commit-msg"]
    # Later releases accept any "Merge ..." message and types in any case; the guide's version notes say why not.
    assert hook["rev"] == "v4.0.0", "the commit-msg hook's rules are the ones CONTRIBUTING.md describes for v4.0.0"
    hook_types = [arg for arg in hook.get("args", []) if not arg.startswith("-")]
    assert types == sorted(hook_types or DEFAULT_COMMIT_TYPES)
    title_types = str((title_check().get("with") or {}).get("types", "")).split()
    assert types == sorted(title_types or DEFAULT_PR_TITLE_TYPES)


def check_subject(subject: str, types: set[str]) -> None:
    """A subject as the guide asks for it: a type, an optional ``!``, ``: `` and a sentence-case summary."""

    match = re.fullmatch(r"(?P<type>[a-z]+)!?: (?P<summary>\S.*)", subject)
    assert match and match["type"] in types, f"{subject!r} is not a Conventional Commit subject without a scope"
    summary = match["summary"]
    if not summary.startswith("<"):  # a placeholder, such as <topic>
        assert summary[0].isupper() and not summary.endswith("."), f"{subject!r}: sentence case, no final period"


def example_messages() -> list[list[str]]:
    """The lines of each example in the guide's commit section: a whole message, or a list of subjects."""

    start = TEXT.index("## Commits and pull requests")
    section = TEXT[start : TEXT.index("\n## ", start)]
    return [code.splitlines() for language, code in code_blocks(section) if language == "text"]


def example_subjects() -> list[str]:
    """Every example subject: in the guide's examples, and quoted in running text here and in the pages that point
    here."""

    types = {row[0].strip("`") for row in table("| Type | For |")}
    messages = example_messages()
    subjects = [lines[0] for lines in messages if len(lines) > 1 and not lines[1]]  # whole messages
    subjects += [line for lines in messages if len(lines) == 1 or lines[1] for line in lines]  # lists of subjects
    pattern = rf"`((?:{'|'.join(sorted(types))})!?: [^`]+)`"
    for page in (GUIDE, REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))):
        subjects += re.findall(pattern, prose(page.read_text()))
    return subjects


def test_the_example_commit_messages_follow_the_rules():
    types = {row[0].strip("`") for row in table("| Type | For |")}
    examples = example_messages()
    assert len(examples) >= 3
    for lines in examples:
        if len(lines) > 1 and not lines[1]:  # a whole message: subject, blank line, body
            breaking = any(line.startswith("BREAKING CHANGE: ") for line in lines[2:])
            assert breaking == ("!" in lines[0].split(":")[0]), f"{lines[0]!r}: `!` goes with a BREAKING CHANGE footer"
    subjects = example_subjects()
    assert len(subjects) > len(examples)
    for subject in subjects:
        check_subject(subject, types)


def test_the_title_check_holds_titles_to_the_guide_s_form():
    options = title_check().get("with") or {}
    assert options.get("disallowScopes") == ".+", "the guide asks for no scope"
    assert options.get("validateSingleCommit") is True, "a single commit's subject becomes the squashed subject"
    summary = re.compile(options["subjectPattern"])
    for subject in example_subjects():
        text = subject.split(": ", 1)[1]
        if not text.startswith("<"):  # a placeholder, such as <the reverted commit's summary>
            assert summary.fullmatch(text), f"the title check would reject the guide's example {subject!r}"
    for text in ("add a thing", "Add a thing.", "`codeact` crashes"):
        assert not summary.fullmatch(text), f"the title check would accept {text!r}"


def test_the_pull_request_template_asks_for_what_the_guide_asks():
    template = (REPO_ROOT / ".github/pull_request_template.md").read_text()
    assert "](.github/pull_request_template.md)" in TEXT
    for heading in ("Summary", "Why", "Links", "Tests", "Checklist"):
        assert f"\n## {heading}\n" in template
    # The title rule, both ways of linking work (as the guide and the RFC process word them) and breaking changes.
    for needed in ("Conventional Commit", "Fixes #", "Implements RFC", "milestone", "BREAKING CHANGE:"):
        assert needed in template, f"the pull request template does not mention {needed!r}"
    assert "`uvx pre-commit run --all-files`" in template and "uvx pre-commit run --all-files" in TEXT
    ty = "uvx pre-commit run ty --hook-stage manual --all-files"
    assert f"`{ty}`" in template and ty in TEXT, "the template asks for the type check the guide asks for"
    assert "`<type>: <summary>`" in template and "`<type>: <summary>`" in TEXT
    for rule in ("no scope", "sentence", "final period"):
        assert rule in template, f"the template does not state the title rule {rule!r}"


def test_every_short_setup_installs_the_git_hooks():
    """The short setups in the README and the install guide install the hooks, as the guide's setup does."""

    for page in (REPO_ROOT / "README.md", REPO_ROOT / "docs/getting-started/install.md"):
        setups = [code for language, code in code_blocks(page.read_text()) if "uv sync --dev" in code]
        assert setups and all("pre-commit install" in code for code in setups), f"{page.name} skips the git hooks"


def test_the_ruff_settings_are_documented():
    ruff = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["ruff"]
    assert f"line-length = {ruff['line-length']}" in TEXT
    rules = ", ".join(f'"{rule}"' for rule in ruff["lint"]["select"])
    assert f"select = [{rules}]" in TEXT


# --------------------------------------------------------------------------- snippets


def test_every_snippet_parses():
    for language, code in code_blocks():
        if language == "python":
            compile(code, f"CONTRIBUTING.md: {code.splitlines()[0]}", "exec")
        elif language == "toml":
            tomllib.loads(code)
        elif language == "yaml":
            yaml.safe_load(code)


@pytest.fixture
def snippet_package(tmp_path, monkeypatch):
    """Write ``path -> code`` files under a temporary directory on ``sys.path``; drop their modules afterwards."""

    monkeypatch.syspath_prepend(str(tmp_path))
    written: list[str] = []

    def write(path: str, code: str) -> Path:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        init = target.parent / "__init__.py"
        if target.name != "__init__.py" and not init.exists():
            init.write_text("")
        target.write_text(code)
        written.append(target.parent.name)
        return target

    yield write
    importlib.invalidate_caches()
    for name in [name for name in sys.modules if name.split(".")[0] in written]:
        del sys.modules[name]


def test_the_environment_and_its_test_run(snippet_package):
    snippet_package("my_envs/repeat.py", block("# my_envs/repeat.py"))
    namespace: dict = {"__name__": "contributing_test_repeat"}
    exec(compile(block("# tests/test_repeat.py"), "tests/test_repeat.py", "exec"), namespace)

    namespace["test_the_right_word_scores_1_and_a_wrong_one_0"]()

    async def task_ids():
        from platoon.envs import get_service

        return await get_service("python://my_envs.repeat:RepeatAfterMe").list_tasks("train")

    assert "train:0" in asyncio.run(task_ids())  # the guide says tasks are numbered <split>:<index>


def test_env_list_prints_what_the_guide_shows(tmp_path, monkeypatch, capsys):
    [config] = [code for language, code in code_blocks() if language == "yaml" and "name: repeat" in code]
    (tmp_path / "repeat.yaml").write_text(config)
    monkeypatch.chdir(tmp_path)

    cli.main(["env", "list", "repeat.yaml"])

    out = capsys.readouterr().out
    expected = block("Environments of repeat.yaml:")
    assert out[out.index("Environments of repeat.yaml:") :].strip() == expected.strip()


def test_the_function_harness_registers_runs_and_shows(snippet_package, capsys):
    from platoon.harness import function
    from platoon.harness.registry import ENTRY_POINT_GROUP, get_harness

    snippet_package("my_harnesses/one_shot.py", block("# my_harnesses/one_shot.py"))
    try:
        module = importlib.import_module("my_harnesses.one_shot")
        record = get_harness("my_harnesses.one_shot:one_shot")
        assert record.ref == "my-one-shot@1"
        assert get_harness("my-one-shot") is record

        class StandIn:
            instruction = "Say hi."

            async def chat(self, messages, **kwargs):
                assert messages == [{"role": "user", "content": "Say hi."}]
                return SimpleNamespace(text="hi")

        result = asyncio.run(module.one_shot(StandIn()))
        assert (result.status, result.final_message) == ("completed", "hi")

        cli.main(["harness", "show", "my_harnesses.one_shot:one_shot"])
        assert capsys.readouterr().out.strip() == block("my-one-shot@1  (function)").strip()
    finally:
        with function._lock:
            function._function_harnesses.pop("my-one-shot@1", None)

    [entry] = [code for language, code in code_blocks() if language == "toml" and "entry-points" in code]
    entry_points = tomllib.loads(entry)["project"]["entry-points"]
    assert list(entry_points) == [ENTRY_POINT_GROUP]
    # One entry point per harness, named as configs name it (platoon imports it only when one does).
    assert entry_points[ENTRY_POINT_GROUP] == {"my-one-shot": "my_harnesses.one_shot:one_shot"}
    [config] = [code for language, code in code_blocks() if language == "yaml" and "my-one-shot" in code]
    [environment] = yaml.safe_load(config)["environments"]
    assert (environment["harness"], environment["package"]) == ("my-one-shot", "my_harnesses.one_shot")


def test_the_toml_spec_is_a_valid_harness_spec():
    from platoon.harness.spec import parse_spec

    spec = parse_spec(tomllib.loads(block("# my-cli-agent.toml")))
    assert spec.ref == "my-cli-agent@0.3.1"


def test_the_renderer_registers_and_takes_renderer_kwargs(snippet_package):
    from tinker_cookbook.renderers import get_renderer, unregister_renderer

    snippet_package("my_renderers/__init__.py", block("# my_renderers/__init__.py"))
    try:
        module = importlib.import_module("my_renderers")
        # Instantiating it proves the snippet implements every abstract method of tinker-cookbook's Renderer.
        renderer = get_renderer("my_model", SimpleNamespace(name_or_path="my-org/my-model"))
        assert isinstance(renderer, module.MyModelRenderer)
        assert renderer.enable_thinking is True  # an attribute, so renderer_kwargs can set it
    finally:
        unregister_renderer("my_model")


def test_the_renderer_names_are_current():
    from platoon.train.tinker.renderers import ALIASES, RENDERER_NAME

    assert f"renderer_name: {RENDERER_NAME}" in TEXT
    presets = {name for name in ALIASES if not name.endswith("_disable_thinking")}
    assert all(f"{name}_disable_thinking" in ALIASES for name in presets)
    for name in presets:
        assert f"`{name}`" in TEXT
    assert "`_disable_thinking` variants" in TEXT


def test_harness_list_rows_match_the_registry():
    """The guide's expected ``platoon harness list`` rows are the built-in harnesses, as the registry lists them."""

    from platoon.harness.registry import available_harnesses, get_harness

    header = "harness                 kind      distribution   source"
    rows = [line.split() for line in block(header).splitlines()[1:]]
    missing = sorted({ref for ref, *_ in rows} - set(available_harnesses()))
    assert not missing, f"CONTRIBUTING.md lists harnesses the registry does not have: {missing}"
    for ref, kind, distribution, _version, source in rows:
        harness = get_harness(ref)
        assert kind == ("spec" if hasattr(harness, "launch") else "function")
        assert distribution == "platoon"
        assert str(harness.source).endswith(source)


# --------------------------------------------------------------------------- references


def test_the_code_the_guide_names_exists():
    references = {
        "platoon.envs.service": ("_BUILTIN_SERVICES", "EnvironmentService", "EnvSession", "Outcome", "ToolSpec"),
        "platoon.cli": ("SERVICE_SCHEMES",),
        "platoon.harness.registry": ("BUILTIN_FUNCTION_MODULES",),
        "platoon.train.tinker.renderers": ("RENDERER_FACTORIES", "ChatTemplateRenderer", "register"),
        "platoon.gym": ("as_environment", "loop"),
    }
    for module_name, names in references.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name}.{name} is gone"
            assert f"`{name}`" in TEXT or f"`{name}(" in TEXT or f"`pt.gym.{name}`" in TEXT, f"{name} is not named"
    assert hasattr(importlib.import_module("platoon.envs.service").Outcome, "infra")


def test_the_paths_and_links_the_guide_names_exist():
    text = prose()
    paths = set(re.findall(r"`((?:platoon|tests|extensions|recipes|docs|examples)/[\w./-]*)`", text))
    paths |= {path for path in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) if "://" not in path}
    paths |= {".pre-commit-config.yaml", ".github/workflows/ci.yml", "pyproject.toml", "ty.toml"}  # read above
    assert paths
    missing = sorted(path for path in paths if not (REPO_ROOT / path).exists())
    assert not missing, f"CONTRIBUTING.md names paths that do not exist: {missing}"


def test_the_in_page_links_resolve():
    anchors = {
        re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        for heading in re.findall(r"^#{2,6} (.+)$", prose(), re.M)
    }
    targets = set(re.findall(r"\]\(#([^)]+)\)", TEXT))
    assert targets and not targets - anchors, f"links to missing sections: {sorted(targets - anchors)}"
