from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
import textwrap
from pathlib import Path

import pytest

from platoon.harness.registry import (
    BUILTIN_SPECS_DIR,
    HarnessRegistry,
    available_harnesses,
    default_registry,
    get_harness,
    get_harness_spec,
    harness_load_failures,
)
from platoon.harness.spec import HarnessSpecError, RenderContext, load_spec, parse_spec

CONTEXT = RenderContext(
    instruction="Fix {state_dir} and ${HOME}",
    base_url="http://127.0.0.1:8000/",
    api_key="secret-key",
    model="policy",
    workdir="/app",
    state_dir="/tmp/agent",
)


def minimal(**overrides) -> dict:
    data = {"name": "toy", "version": "1.0", "launch": {"argv": ["toy", "{instruction}"]}}
    data.update(overrides)
    return data


def write_spec(directory: Path, body: str, name: str = "toy.toml") -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(body))
    return path


def test_builtin_specs_load_and_resolve_by_name_version_and_prefix():
    assert get_harness_spec("platoon-bash").ref == "platoon-bash@1"
    assert get_harness_spec("mini-swe-agent").ref == "mini-swe-agent@2.4.6"
    assert get_harness_spec("mini-swe-agent@2").ref == "mini-swe-agent@2.4.6"
    assert get_harness_spec("mini-swe-agent@2.4.6").ref == "mini-swe-agent@2.4.6"
    with pytest.raises(ValueError, match="Unknown harness 'mini-swe-agent@3'"):
        get_harness_spec("mini-swe-agent@3")
    with pytest.raises(ValueError, match="Unknown harness 'nope'"):
        get_harness_spec("nope")


def test_mini_swe_agent_points_litellm_at_the_gateway_and_keeps_the_secret_in_env():
    spec = get_harness_spec("mini-swe-agent")
    rendered = spec.render(CONTEXT)

    assert rendered.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert rendered.env["OPENAI_API_BASE"] == "http://127.0.0.1:8000/v1"
    assert rendered.env["OPENAI_API_KEY"] == "secret-key"
    assert all("secret-key" not in arg for arg in rendered.argv)
    argv = list(rendered.argv)
    assert argv[argv.index("--model") + 1] == "openai/policy"
    assert argv[argv.index("--task") + 1] == CONTEXT.instruction
    assert {"--yolo", "--exit-immediately"} <= set(argv)
    assert rendered.final_message_file == "/tmp/agent/trajectory.json"
    assert spec.final_message.json_key == "info.submission"
    assert spec.install.commands and "uv tool install" in spec.install.commands[0]


def test_mini_swe_agent_installs_uv_with_pip_when_the_image_has_no_downloader(tmp_path):
    """python:3.12-slim task images have pip but neither curl nor wget, so uv comes from pip."""

    home, image_bin, calls = tmp_path / "home", tmp_path / "bin", tmp_path / "calls"
    local_bin = home / ".local" / "bin"
    image_bin.mkdir()
    # Fakes that record each call and install what the real command would into ~/.local/bin, offline.
    record = f'#!/bin/sh\necho "${{0##*/}} $*" >> {calls}\n'
    mini_swe_agent = record
    uv = record + (
        f'if [ "$1" = tool ]; then printf %s {shlex.quote(mini_swe_agent)} > {local_bin}/mini-swe-agent; '
        f"/bin/chmod +x {local_bin}/mini-swe-agent; fi\n"
    )
    python3 = (
        record
        + f"/bin/mkdir -p {local_bin}; printf %s {shlex.quote(uv)} > {local_bin}/uv; /bin/chmod +x {local_bin}/uv\n"
    )
    (image_bin / "python3").write_text(python3)
    (image_bin / "python3").chmod(0o755)

    (command,) = get_harness_spec("mini-swe-agent").install.commands
    result = subprocess.run(
        ["/bin/bash", "-c", command], env={"HOME": str(home), "PATH": str(image_bin)}, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == [
        "python3 -m pip install --user --quiet uv",
        "uv python install 3.12",
        "uv tool install --python 3.12 mini-swe-agent==2.4.6 --with litellm --with orjson --with fastapi",
        "mini-swe-agent --help",
    ]


def test_platoon_bash_ships_its_script_as_an_asset():
    spec = get_harness_spec("platoon-bash")
    rendered = spec.render(CONTEXT)

    (script,) = rendered.assets.values()
    assert script == (BUILTIN_SPECS_DIR.parent / "platoon_bash.py").resolve()
    assert rendered.files == {"instruction.md": CONTEXT.instruction}
    assert rendered.env == {"PLATOON_API_KEY": "secret-key"}
    assert spec.exit_status(0) == "completed"
    assert spec.exit_status(3) == "completed"
    assert spec.exit_status(2) == "error"
    assert spec.exit_status(None) == "error"


def test_rendering_substitutes_once_and_leaves_other_braces_literal():
    spec = parse_spec(
        minimal(
            launch={
                "argv": ["sh", "-c", 'echo "${HOME}" {model}', "{instruction}"],
                "env": {"KEY": "{api_key}", "URL": "{base_url}/v1"},
                "files": {"{state_dir}/config.json": '{"model": "{model}", "cwd": "{workdir}"}'},
            }
        )
    )
    rendered = spec.render(CONTEXT)

    assert rendered.argv == ("sh", "-c", 'echo "${HOME}" policy', "Fix {state_dir} and ${HOME}")
    assert rendered.env == {"KEY": "secret-key", "URL": "http://127.0.0.1:8000/v1"}
    assert rendered.files == {"/tmp/agent/config.json": '{"model": "policy", "cwd": "/app"}'}


def test_builtins_disable_knob_appends_args_and_env():
    spec = parse_spec(
        minimal(builtins={"tools": ["shell"], "disable": {"args": ["--no-shell"], "env": {"TOOLS": "none"}}})
    )

    assert spec.render(CONTEXT).argv == ("toy", CONTEXT.instruction)
    disabled = spec.render(CONTEXT, disable_builtins=True)
    assert disabled.argv == ("toy", CONTEXT.instruction, "--no-shell")
    assert disabled.env == {"TOOLS": "none"}
    with pytest.raises(HarnessSpecError, match="no knob to disable"):
        parse_spec(minimal()).render(CONTEXT, disable_builtins=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"surprise": 1}, r"unknown key\(s\) \['surprise'\]"),
        ({"launch": {"argv": ["toy"], "cwd": "/"}}, r"unknown key\(s\) \['cwd'\]"),
        ({"launch": {"argv": []}}, "argv must not be empty"),
        ({"launch": {"argv": ["toy", "{instrution}"]}}, r"unknown placeholder\(s\) \['instrution'\]"),
        ({"launch": {"argv": ["toy", "--key={api_key}"]}}, r"may only be rendered into \[launch.env\]"),
        ({"launch": {"argv": ["toy"], "files": {"a": "{api_key}"}}}, r"may only be rendered into \[launch.env\]"),
        ({"launch": {"argv": ["{toolchain}/bin/toy"]}}, r"unknown placeholder\(s\) \['toolchain'\]"),
        ({"dialect": "gemini"}, "dialect 'gemini' is not one of"),
        ({"version": 2}, "version must be a string"),
        ({"name": "bad name"}, "name 'bad name' must match"),
        ({"timeout_s": 0}, "timeout_s must be a positive number"),
        ({"final_message": {}}, "set exactly one of file or stdout_regex"),
        ({"final_message": {"file": "a", "stdout_regex": "b"}}, "set exactly one of file or stdout_regex"),
        ({"final_message": {"stdout_regex": "x", "json_key": "a"}}, "json_key requires file"),
        ({"final_message": {"stdout_regex": "("}}, "invalid stdout_regex"),
        ({"exit_codes": {"three": "completed"}}, "is not an integer exit code"),
        ({"exit_codes": {"3": "limit"}}, "maps to 'limit'"),
        ({"builtins": {"disable": {}}}, "disable must set args or env"),
        ({"install": {"toolchain": "relative/dir"}}, "toolchain 'relative/dir' must be an absolute path or a name"),
        ({"launch": {"argv": ["toy"], "assets": {"x.py": "missing.py"}}}, "asset 'missing.py' not found"),
    ],
)
def test_invalid_specs_are_rejected_with_a_located_message(overrides, message):
    with pytest.raises(HarnessSpecError, match=message):
        parse_spec(minimal(**overrides), base_dir=Path(__file__).parent)


def test_toolchain_placeholder_and_install_commands():
    spec = parse_spec(
        minimal(
            install={"toolchain": "/opt/toolchains/toy", "commands": ["echo ${HOME} {not-a-template}"]},
            launch={"argv": ["{toolchain}/bin/toy"]},
        )
    )
    assert spec.render(dataclasses.replace(CONTEXT, toolchain="/opt/toolchains/toy")).argv == (
        "/opt/toolchains/toy/bin/toy",
    )
    assert spec.install.commands == ("echo ${HOME} {not-a-template}",)


def test_load_spec_reports_toml_errors_with_the_path(tmp_path):
    path = write_spec(tmp_path, "name = [")
    with pytest.raises(HarnessSpecError, match="toy.toml: invalid TOML"):
        load_spec(path)


def test_registry_loads_directories_rejects_duplicates_and_accepts_entry_point_values(tmp_path):
    write_spec(tmp_path, 'name = "toy"\nversion = "1.0"\n[launch]\nargv = ["toy"]\n', "a.toml")
    write_spec(tmp_path, 'name = "toy"\nversion = "1.1"\n[launch]\nargv = ["toy"]\n', "b.toml")
    registry = HarnessRegistry()
    registry.add_path(tmp_path)

    assert registry.refs() == ["toy@1.0", "toy@1.1"]
    assert registry.get("toy@1.1").version == "1.1"
    with pytest.raises(ValueError, match="ambiguous"):
        registry.get("toy")
    with pytest.raises(ValueError, match="ambiguous"):
        registry.get("toy@1")
    with pytest.raises(ValueError, match="defined twice"):
        registry.add_path(tmp_path / "a.toml")

    registry.add_entry_point_value([parse_spec(minimal(name="other")), str(BUILTIN_SPECS_DIR / "platoon-bash.toml")])
    assert registry.get("other").ref == "other@1.0"
    assert registry.get("platoon-bash").ref == "platoon-bash@1"
    with pytest.raises(TypeError, match="must name a spec path or HarnessSpec"):
        registry.add_entry_point_value(42)


@pytest.fixture
def broken_entry_point(tmp_path, monkeypatch):
    """An installed distribution whose ``platoon.harnesses`` entry point raises on load, as a harness package
    built against renamed platoon APIs does."""

    from platoon import extensions

    dist_info = tmp_path / "broken_harnesses-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: broken-harnesses\nVersion: 0.1\n")
    (dist_info / "entry_points.txt").write_text("[platoon.harnesses]\nbroken = broken_harness_module\n")
    (tmp_path / "broken_harness_module.py").write_text("raise ImportError('cannot import name NodeContext')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    default_registry.cache_clear()
    extensions.refresh()
    yield "broken = broken_harness_module"
    monkeypatch.undo()
    default_registry.cache_clear()
    extensions.refresh()


def test_a_broken_entry_point_leaves_every_other_harness_available(broken_entry_point, capsys):
    from platoon import cli

    assert get_harness("platoon-codeact").name == "platoon-codeact"
    assert get_harness_spec("platoon-bash").ref == "platoon-bash@1"
    assert {"platoon-bash@1", "mini-swe-agent@2.4.6"} <= set(available_harnesses())
    # Looking up other harnesses never imports it: only its own name, or a name nothing else has, does.
    assert broken_entry_point not in harness_load_failures()
    # A missing harness may be the broken package's: the errors say what failed to load.
    with pytest.raises(ValueError, match=f"failed to load: .*{broken_entry_point}"):
        get_harness("acme-openhands")
    with pytest.raises(ValueError, match=f"failed to load: .*{broken_entry_point}"):
        get_harness_spec("acme-openhands")
    assert harness_load_failures()[broken_entry_point] == "ImportError: cannot import name NodeContext"
    with pytest.raises(ValueError, match="broken = broken_harness_module \\(broken-harnesses 0.1\\) failed to load"):
        get_harness("broken")

    cli.main(["harness", "list"])
    out, err = capsys.readouterr()
    assert any(line.split()[:2] == ["platoon-bash@1", "spec"] for line in out.splitlines())
    assert any(
        line.split() == ["broken", "-", "broken-harnesses", "0.1", "broken_harness_module"] for line in out.splitlines()
    )
    assert f"{broken_entry_point} failed to load" in err


def test_get_harness_spec_accepts_a_spec_path(tmp_path):
    path = write_spec(tmp_path, 'name = "toy"\nversion = "9"\n[launch]\nargv = ["toy"]\n')
    assert get_harness_spec(str(path)).ref == "toy@9"


def test_max_steps_comes_from_the_run_and_a_missing_budget_is_an_error():
    spec = parse_spec(minimal(launch={"argv": ["toy", "--steps", "{max_steps}"]}))

    assert spec.render(dataclasses.replace(CONTEXT, max_steps=12)).argv == ("toy", "--steps", "12")
    with pytest.raises(HarnessSpecError, match=r"\{max_steps\}"):
        spec.render(CONTEXT)


def test_spec_max_steps_is_a_whole_positive_default():
    assert parse_spec(minimal(max_steps=40)).max_steps == 40
    with pytest.raises(HarnessSpecError, match="whole number"):
        parse_spec(minimal(max_steps=2.5))
    with pytest.raises(HarnessSpecError):
        parse_spec(minimal(max_steps=0))


def test_builtin_specs_bound_their_steps_by_the_run_budget():
    rendered = {
        name: get_harness_spec(name).render(dataclasses.replace(CONTEXT, max_steps=9))
        for name in ("mini-swe-agent", "platoon-bash")
    }

    assert "agent.step_limit=9" in rendered["mini-swe-agent"].argv
    assert rendered["platoon-bash"].argv[-2:] == ("--max-turns", "9")
    assert get_harness_spec("mini-swe-agent").max_steps and get_harness_spec("platoon-bash").max_steps


MCP_CONTEXT = dataclasses.replace(CONTEXT, mcp_url="http://127.0.0.1:8000/mcp")


def test_mcp_table_applies_only_when_the_agent_serves_tools():
    spec = get_harness_spec("platoon-bash")

    assert spec.mcp.enabled
    assert "--mcp-url" not in spec.render(CONTEXT).argv
    assert spec.render(MCP_CONTEXT).argv[-2:] == ("--mcp-url", "http://127.0.0.1:8000/mcp")
    disabled = parse_spec(minimal(mcp={"enabled": False, "args": ["--mcp", "{mcp_url}"]}))
    assert disabled.render(MCP_CONTEXT).argv == ("toy", CONTEXT.instruction)


def test_mcp_config_template_is_written_to_the_state_dir():
    spec = parse_spec(
        minimal(
            mcp={
                "config_template": '{"mcpServers": {"env": {"url": "{mcp_url}", "token": "${PLATOON_API_KEY}"}}}',
                "config_file": "config/mcp.json",
                "args": ["--mcp-config", "{mcp_config_path}"],
                "env": {"MCP_TOKEN": "{api_key}", "MCP_URL": "{mcp_url}"},
            }
        )
    )

    rendered = spec.render(MCP_CONTEXT)

    assert json.loads(rendered.files["config/mcp.json"]) == {
        "mcpServers": {"env": {"url": "http://127.0.0.1:8000/mcp", "token": "${PLATOON_API_KEY}"}}
    }
    assert rendered.argv[-2:] == ("--mcp-config", "/tmp/agent/config/mcp.json")
    assert rendered.env == {"MCP_TOKEN": "secret-key", "MCP_URL": "http://127.0.0.1:8000/mcp"}
    assert spec.render(CONTEXT).files == {}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"launch": {"argv": ["toy", "{mcp_url}"]}}, "unknown placeholder"),
        ({"mcp": {"args": ["--config", "{mcp_config_path}"]}}, "unknown placeholder"),
        ({"mcp": {"args": ["--key", "{api_key}"]}}, "may only be rendered"),
        ({"mcp": {"config_template": "{api_key}"}}, "may only be rendered"),
        ({"mcp": {"enabled": "yes"}}, "true or false"),
        ({"mcp": {"url": "{mcp_url}"}}, "unknown key"),
    ],
)
def test_mcp_placeholders_render_only_into_the_mcp_table(overrides, message):
    with pytest.raises(HarnessSpecError, match=message):
        parse_spec(minimal(**overrides))
