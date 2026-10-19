"""Build plans of Harbor task Dockerfiles: the shared build prefix, its key, and the deferred suffix."""

from __future__ import annotations

import gzip
import io
import socket
import tarfile
from pathlib import Path

import pytest

from platoon.envs.harbor.dockerfile import DockerfileError, expand, parse_dockerfile, plan_dockerfile, remote_source

SMOLDATA = """FROM docker.io/savatar101/env-data-agent-train:base
WORKDIR /workdir
COPY input/ /data/input/
RUN pip install --no-cache-dir math-verify==0.9.0
"""
SMOLDATA_REORDERED = """FROM docker.io/savatar101/env-data-agent-train:base
WORKDIR /workdir
RUN pip install --no-cache-dir math-verify==0.9.0
COPY input/ /data/input/
"""
MCO = """FROM python:3.11.16-slim
RUN apt-get update && apt-get install -y --no-install-recommends tmux curl git procps build-essential \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.4.6 scipy==1.17.1 sympy==1.14.0 networkx==3.6.1 \\
    python-sat==1.9.dev15 z3-solver==5.1.0.0 PuLP==3.3.2 ortools==9.15.6755
WORKDIR /workdir
COPY instance.json /workdir/instance.json
"""
CHECKSUM = "sha256:" + "3e" * 32
PDG = "https://raw.githubusercontent.com/org/repo/0123456789abcdef0123456789abcdef01234567"
# The verifier Dockerfile of nuclear-fusion-rl's b03, cut down: files fetched at build time through an ARG, each
# checked by its SHA-256, a data stage that unpacks a remote archive, and a last stage that declares the ARG again.
FETCHING_VERIFIER = f"""# syntax=docker/dockerfile:1.7
FROM python:3.11.6-slim AS base
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ARG PDG={PDG}
ADD --checksum={CHECKSUM} ${{PDG}}/pkg/__init__.py /opt/pkg/__init__.py

FROM base AS data
ADD --checksum={CHECKSUM} \\
    https://zenodo.org/records/1/files/data.tar.gz?download=1 /tmp/data.tar.gz
RUN mkdir -p /out && tar -xzf /tmp/data.tar.gz -C /out

FROM base
ARG PDG={PDG}
ADD --checksum={CHECKSUM} ${{PDG}}/tmp/checkpoint /verifier_data/checkpoint
COPY --from=data /out /verifier_data/out
COPY grader.py /tests/grader.py
RUN chmod 700 /verifier_data
ENV GRADER_DATA_DIR=/verifier_data
"""


def _context(root: Path, dockerfile: str, files: dict[str, str | bytes] | None = None) -> Path:
    context = root / "environment"
    context.mkdir(parents=True, exist_ok=True)
    (context / "Dockerfile").write_text(dockerfile)
    for relative, data in (files or {}).items():
        path = context / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data) if isinstance(data, bytes) else path.write_text(data)
    return context


def test_parser_handles_continuations_comments_directives_and_heredocs():
    parsed = parse_dockerfile(
        "# syntax=docker/dockerfile:1\n# escape=\\\n\n# a comment\nfrom python:3.12 AS base\n"
        "RUN apt-get update && \\\n    # dropped inside a continuation\n\n    apt-get install -y curl\n"
        "RUN <<EOF\necho one\necho two\nEOF\nRUN echo $((1<<2)) && cat <<<'here-string'\n"
        "COPY <<-END /etc/motd\n\thello\n\tEND\n"
        'CMD ["python", "-c", "print(1)"]\n'
    )
    assert parsed.directives == ("# syntax=docker/dockerfile:1", "# escape=\\")
    keywords = [instruction.keyword for instruction in parsed.instructions]
    assert keywords == ["FROM", "RUN", "RUN", "RUN", "COPY", "CMD"]
    assert parsed.instructions[1].arguments == "apt-get update &&     apt-get install -y curl"
    assert "dropped" not in parsed.instructions[1].text
    assert parsed.instructions[2].heredocs == ("echo one\necho two",)
    assert parsed.instructions[3].heredocs == ()
    assert parsed.instructions[4].heredocs == ("hello",)
    assert parsed.instructions[5].flags_and_operands() == ({}, ["python", "-c", "print(1)"])
    with pytest.raises(DockerfileError, match="not terminated"):
        parse_dockerfile("FROM x\nRUN <<EOF\necho\n")


def test_backtick_escape_directive():
    parsed = parse_dockerfile("# escape=`\nFROM x\nRUN echo a `\n    b\n")
    assert parsed.instructions[1].arguments == "echo a     b"


def test_the_trailing_copy_is_deferred_and_tasks_share_a_key(tmp_path: Path):
    first = plan_dockerfile(_context(tmp_path / "a", SMOLDATA_REORDERED, {"input/a.csv": "1"}) / "Dockerfile")
    second = plan_dockerfile(_context(tmp_path / "b", SMOLDATA_REORDERED, {"input/b.csv": "2"}) / "Dockerfile")
    assert first.key == second.key and first.key.startswith("sha256:")
    assert first.deferred == (("COPY", "input/", "/data/input/"),)
    assert [instruction.keyword for instruction in first.prefix] == ["FROM", "WORKDIR", "RUN"]
    assert first.base_images == ("docker.io/savatar101/env-data-agent-train:base",)
    assert first.context_files == ()
    assert first.prefix_dockerfile() == (
        "FROM docker.io/savatar101/env-data-agent-train:base\nWORKDIR /workdir\n"
        "RUN pip install --no-cache-dir math-verify==0.9.0\n"
    )


def test_a_copy_before_the_last_run_makes_the_key_per_task(tmp_path: Path):
    first = plan_dockerfile(_context(tmp_path / "a", SMOLDATA, {"input/a.csv": "1"}) / "Dockerfile")
    second = plan_dockerfile(_context(tmp_path / "b", SMOLDATA, {"input/a.csv": "2"}) / "Dockerfile")
    same = plan_dockerfile(_context(tmp_path / "c", SMOLDATA, {"input/a.csv": "1"}) / "Dockerfile")
    assert first.deferred == ()
    assert first.context_files == ("input",)
    assert first.key != second.key  # the inputs are in the build
    assert first.key == same.key  # same bytes, same image


def test_math_tasks_defer_workdir_and_copy(tmp_path: Path):
    plan = plan_dockerfile(_context(tmp_path, MCO, {"instance.json": "{}"}) / "Dockerfile")
    assert plan.deferred == (("WORKDIR", "/workdir"), ("COPY", "instance.json", "/workdir/instance.json"))
    assert [instruction.keyword for instruction in plan.prefix] == ["FROM", "RUN", "RUN"]
    other = plan_dockerfile(_context(tmp_path / "other", MCO, {"instance.json": '{"n": 3}'}) / "Dockerfile")
    assert other.key == plan.key


def test_prefix_context_files_enter_the_key(tmp_path: Path):
    dockerfile = "FROM python:3.12-slim\nCOPY generate_inputs.py /opt/\nRUN python /opt/generate_inputs.py\n"
    first = plan_dockerfile(_context(tmp_path / "a", dockerfile, {"generate_inputs.py": "seed = 1"}) / "Dockerfile")
    second = plan_dockerfile(_context(tmp_path / "b", dockerfile, {"generate_inputs.py": "seed = 2"}) / "Dockerfile")
    assert first.context_files == ("generate_inputs.py",)
    assert first.key != second.key
    with pytest.raises(DockerfileError, match="missing"):
        plan_dockerfile(_context(tmp_path / "c", dockerfile) / "Dockerfile")


def test_the_executable_bit_of_a_prefix_file_enters_the_key(tmp_path: Path):
    dockerfile = "FROM x\nCOPY run.sh /opt/\nRUN /opt/run.sh\n"
    context = _context(tmp_path / "a", dockerfile, {"run.sh": "#!/bin/sh\n"})
    before = plan_dockerfile(context / "Dockerfile").key
    (context / "run.sh").chmod(0o755)
    assert plan_dockerfile(context / "Dockerfile").key != before


@pytest.mark.parametrize(
    ("suffix", "why"),
    [
        ("COPY --from=build /out /bin/", "--from"),
        ("COPY --chmod=755 run.sh /opt/", "--chmod"),
        ("ADD https://example.com/x.tgz /opt/", "URL"),
        ("COPY *.json /workdir/", "wildcard"),
        ("COPY $SRC /workdir/", "variable"),
        ("ADD data.tar.gz /data/", "archive"),
        ("COPY <<EOF /etc/x\nhi\nEOF", "here-document"),
    ],
)
def test_steps_that_cannot_be_deferred_stay_in_the_build(tmp_path: Path, suffix: str, why: str):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo("x")
        tar.addfile(info, io.BytesIO(b""))
    context = _context(
        tmp_path,
        f"FROM x\nRUN true\n{suffix}\nWORKDIR /w\n",
        {"run.sh": "#!/bin/sh\n", "a.json": "{}", "data.tar.gz": gzip.compress(archive.getvalue())},
    )
    plan = plan_dockerfile(context / "Dockerfile")
    assert plan.deferred == (("WORKDIR", "/w"),)
    assert plan.prefix[-1].keyword in ("COPY", "ADD")
    assert why in " ".join(plan.notes) or why == "here-document" and "here-document" in " ".join(plan.notes)


def test_a_missing_copy_source_fails_the_plan(tmp_path: Path):
    with pytest.raises(DockerfileError, match="'missing.json' is missing"):
        plan_dockerfile(_context(tmp_path, "FROM x\nRUN true\nCOPY missing.json /workdir/\n") / "Dockerfile")


def test_a_dockerignore_keeps_copies_in_the_build(tmp_path: Path):
    context = _context(tmp_path, "FROM x\nRUN true\nCOPY a.json /w/\n", {"a.json": "{}", ".dockerignore": "*.md\n"})
    plan = plan_dockerfile(context / "Dockerfile")
    assert plan.deferred == () and plan.context_files == (".",)


def test_env_arg_label_and_user_in_the_suffix(tmp_path: Path):
    context = _context(
        tmp_path,
        "ARG BASE=python:3.12\nFROM $BASE\nARG VERSION=1.0\nRUN true\n"
        "ENV PATH=/opt/bin:$PATH MODE='$literal' NAME=\"two words\"\nLABEL org.example=x\nEXPOSE 80\n"
        'ARG LATE\nWORKDIR data\nCOPY a.json b.json ./\nUSER learner\nCMD ["sleep", "1"]\n',
        {"a.json": "{}", "b.json": "{}"},
    )
    plan = plan_dockerfile(context / "Dockerfile")
    assert plan.base_images == ("python:3.12",)
    assert plan.deferred == (
        ("ARG", "VERSION", "1.0"),
        ("ENV", "PATH", "/opt/bin:$PATH"),
        ("ENV", "MODE", "\\$literal"),
        ("ENV", "NAME", "two words"),
        ("ARG", "LATE", ""),
        ("WORKDIR", "data"),
        ("COPY", "a.json", "b.json", "./"),
    )
    assert plan.suffix_user == "learner"


def test_a_dockerfile_without_from_is_refused(tmp_path: Path):
    with pytest.raises(DockerfileError, match="no FROM"):
        plan_dockerfile(_context(tmp_path, "RUN true\n") / "Dockerfile")


def test_multi_stage_builds_keep_their_trailing_copies_deferred(tmp_path: Path):
    context = _context(
        tmp_path,
        "FROM golang AS build\nRUN go build -o /out/app .\nFROM alpine\nCOPY --from=build /out/app /bin/\n"
        "COPY config.json /etc/app/\n",
        {"config.json": "{}"},
    )
    plan = plan_dockerfile(context / "Dockerfile")
    assert plan.deferred == (("COPY", "config.json", "/etc/app/"),)
    assert plan.base_images == ("golang", "alpine")


def test_expand_follows_dockerfile_rules():
    variables = {"A": "", "B": "b", "D": "d"}
    assert expand("${A:-x}/${A-y}/$B/${B}c/\\$C/${D:+set}/${E:+set}/$E.", variables) == "x//b/bc/$C/set//."


@pytest.mark.parametrize(
    ("source", "variables", "kind"),
    [
        ("https://example.com/data.tar.gz", None, "url"),
        ("http://example.com/x", None, "url"),
        ("https://zenodo.org/records/1/files/data.tar.gz?download=1", None, "url"),
        ("https://github.com/org/repo.git", None, "git"),
        ("https://github.com/org/repo.git#v1.0:sub/dir", None, "git"),
        ("git://example.com/repo", None, "git"),
        ("ssh://git@example.com/repo.git", None, "git"),
        ("git@github.com:org/repo.git", None, "git"),
        ("deploy@example.com:repo#main", None, "git"),
        ("github.com/org/repo", None, None),  # a path, as BuildKit reads it
        ("ftp://example.com/x", None, None),  # a path too: only http and https URLs are fetched
        ("data/input.csv", None, None),
        ("${PDG}/x.py", {"PDG": "https://example.com/pkg"}, "url"),
        ("${REPO}#v1", {"REPO": "https://github.com/org/repo.git"}, "git"),
        ("${PDG}/x.py", {}, None),  # a value the planner does not know
        ("https://example.com/${VERSION}/x", {}, "url"),  # a URL whatever the value
    ],
)
def test_remote_sources_are_told_apart_as_buildkit_does(source: str, variables: dict | None, kind: str | None):
    assert remote_source(source, variables) == kind


def test_remote_adds_through_an_arg_stay_in_the_build_and_read_no_context(tmp_path: Path):
    # Before, the planner took ${PDG}/pkg/__init__.py for a context path once it expanded it, and refused the
    # Dockerfile: "line 6: ADD source 'https://raw.githubusercontent.com/...' is missing".
    files: dict[str, str | bytes] = {"requirements.txt": "numpy\n", "grader.py": "print(1)\n", "notes.md": "unread"}
    plan = plan_dockerfile(_context(tmp_path, FETCHING_VERIFIER, files) / "Dockerfile")
    assert plan.context_files == ("grader.py", "requirements.txt")
    assert plan.deferred == (("ARG", "PDG", PDG), ("ENV", "GRADER_DATA_DIR", "/verifier_data"))
    built = plan.prefix_dockerfile()
    assert built.startswith("# syntax=docker/dockerfile:1.7\n")
    assert f"ADD --checksum={CHECKSUM} ${{PDG}}/tmp/checkpoint /verifier_data/checkpoint\n" in built
    assert "https://zenodo.org/records/1/files/data.tar.gz?download=1 /tmp/data.tar.gz" in built


def test_the_key_covers_a_remote_add_as_written_and_planning_reads_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def offline(*args: object, **kwargs: object) -> None:
        raise AssertionError("planning opened a network connection")

    monkeypatch.setattr(socket, "create_connection", offline)
    monkeypatch.setattr(socket.socket, "connect", offline)
    files: dict[str, str | bytes] = {"requirements.txt": "numpy\n", "grader.py": "print(1)\n"}

    def key(name: str, dockerfile: str, extra: dict[str, str | bytes] | None = None) -> str:
        return plan_dockerfile(_context(tmp_path / name, dockerfile, {**files, **(extra or {})}) / "Dockerfile").key

    first = key("a", FETCHING_VERIFIER)
    assert key("b", FETCHING_VERIFIER, {"notes.md": "read by no step"}) == first
    keys = {
        first,
        key("checksum", FETCHING_VERIFIER.replace(CHECKSUM, "sha256:" + "4f" * 32, 1)),
        key("url", FETCHING_VERIFIER.replace(PDG, PDG.replace("0123", "fedc"), 1)),
        key("archive", FETCHING_VERIFIER.replace("records/1/", "records/2/")),
    }
    assert len(keys) == 4


def test_an_add_of_a_url_named_by_an_env_value_stays_in_the_build(tmp_path: Path):
    dockerfile = "FROM x\nENV DATA_URL=https://example.com/data.bin\nADD $DATA_URL /data/data.bin\nRUN true\n"
    plan = plan_dockerfile(_context(tmp_path, dockerfile) / "Dockerfile")
    assert [instruction.keyword for instruction in plan.prefix] == ["FROM", "ENV", "ADD", "RUN"]
    assert plan.context_files == () and plan.deferred == ()


@pytest.mark.parametrize(
    ("dockerfile", "note"),
    [
        ("FROM x\nRUN true\nADD https://example.com/data.tar.gz /data/\nWORKDIR /w\n", "ADD of a URL"),
        ("FROM x\nARG PDG=https://example.com/pkg\nRUN true\nADD ${PDG}/a.py /opt/a.py\nWORKDIR /w\n", "ADD of a URL"),
        ("FROM x\nRUN true\nENV DATA=https://example.com/d.bin\nADD $DATA /data/d.bin\nWORKDIR /w\n", "ADD of a URL"),
        (f"FROM x\nRUN true\nADD --checksum={CHECKSUM} https://example.com/x /opt/x\nWORKDIR /w\n", "ADD --checksum"),
        ("FROM x\nRUN true\nADD https://github.com/org/repo.git#v1.0 /src\nWORKDIR /w\n", "ADD of a git repository"),
        ("FROM x\nRUN true\nADD git@github.com:org/repo.git /src\nWORKDIR /w\n", "ADD of a git repository"),
    ],
)
def test_a_trailing_add_of_remote_content_stays_in_the_build(tmp_path: Path, dockerfile: str, note: str):
    plan = plan_dockerfile(_context(tmp_path, dockerfile) / "Dockerfile")
    assert plan.prefix[-1].keyword == "ADD"  # built, never uploaded by a session
    assert plan.deferred[-1] == ("WORKDIR", "/w") and all(step[0] != "COPY" for step in plan.deferred)
    assert plan.context_files == () and plan.copied == ()
    assert any(note in why for why in plan.notes), plan.notes
    assert not any("archive" in why for why in plan.notes)  # a remote archive is fetched as it is, not extracted


def test_variables_in_sources_take_the_scopes_of_a_build(tmp_path: Path):
    def context_files(name: str, dockerfile: str) -> tuple[str, ...]:
        return plan_dockerfile(_context(tmp_path / name, dockerfile, {"a.json": "{}"}) / "Dockerfile").context_files

    # A stage built FROM another inherits its ARG values, and a stage that declares a global ARG again gets its
    # value: the source is a URL, which no context file serves.
    assert context_files("child", "FROM x AS base\nARG U=https://h/\nFROM base\nADD ${U}a.json /a\nRUN true\n") == ()
    assert context_files("declared", "ARG U=https://h/\nFROM x\nARG U\nADD ${U}a.json /a\nRUN true\n") == ()
    # Another stage's ARG, or a global ARG the stage does not declare, has no value in the stage that the planner
    # knows (the base image may set one), so the whole context enters the key.
    assert context_files("other", "FROM x AS one\nARG U=https://h/\nFROM y\nADD ${U}a.json /a\nRUN true\n") == (".",)
    assert context_files("global", "ARG U=https://h/\nFROM x\nADD ${U}a.json /a\nRUN true\n") == (".",)
    assert context_files("image-env", "FROM x\nENV U=$HOME/\nADD ${U}a.json /a\nRUN true\n") == (".",)
    # An ENV value wins over an ARG of the same name, and the values of one ENV step see the variables before it.
    assert context_files("env-wins", "FROM x\nENV S=a.json\nARG S=https://h/a.json\nADD $S /a\nRUN true\n") == (
        "a.json",
    )
    same_step = "FROM x\nENV S=a.json\nENV S=https://h/a.json T=$S\nADD $T /a\nRUN true\n"
    assert context_files("same-step", same_step) == ("a.json",)


@pytest.mark.parametrize("source", ["https://example.com/x", "${URL}", "git@github.com:org/repo.git"])
def test_a_copy_of_remote_content_fails_the_plan_as_it_fails_a_build(tmp_path: Path, source: str):
    dockerfile = f"FROM x\nARG URL=https://example.com/x\nCOPY {source} /x\nRUN true\n"
    with pytest.raises(DockerfileError, match=r"line 3: COPY cannot fetch .*; only ADD can"):
        plan_dockerfile(_context(tmp_path, dockerfile) / "Dockerfile")


@pytest.mark.parametrize(
    ("first", "files"),
    [
        ("COPY a.txt /a", {"a.txt": "a", ".dockerignore": "*.pyc\n"}),
        ("COPY *.txt /a/", {"a.txt": "a"}),
        ("COPY ${UNSET}a.txt /a", {"a.txt": "a"}),
        ("RUN --mount=type=bind,target=/m true", {}),
    ],
)
def test_a_copy_of_remote_content_fails_the_plan_after_a_step_that_reads_the_whole_context(
    tmp_path: Path, first: str, files: dict[str, str | bytes]
):
    # The step before makes the key cover the whole context; the COPY after it is still read, and still refused.
    context = _context(tmp_path, f"FROM x\n{first}\nCOPY https://example.com/x /x\nRUN true\n", files)
    with pytest.raises(DockerfileError, match=r"line 3: COPY cannot fetch a URL \(https://example\.com/x\)"):
        plan_dockerfile(context / "Dockerfile")
    (context / "Dockerfile").write_text(f"FROM x\n{first}\nADD https://example.com/x /x\nCOPY b.txt /b\nRUN true\n")
    assert plan_dockerfile(context / "Dockerfile").context_files == (".",)  # a missing b.txt is not looked for
