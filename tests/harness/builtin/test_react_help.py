"""``platoon harness show platoon-react`` says what the harness does and which options it takes: the CLI shows the
harness function's docstring, so it must stand on its own."""

from __future__ import annotations

from platoon import cli


def test_harness_show_describes_platoon_react(capsys):
    cli.main(["harness", "show", "platoon-react"])

    out = capsys.readouterr().out
    assert out.startswith("platoon-react@1  (function)\n")
    assert "module docstring" not in out
    for words in ("tool", "terminal tool", "system_prompt", "sampling", "max_completion_tokens", "limit"):
        assert words in out, words
