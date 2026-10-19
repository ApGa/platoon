# Contributing to platoon

Thank you for helping. This guide covers the development setup, the checks that CI runs, the commit and pull
request conventions, and how to add the three things contributors add most often: an environment, a harness and
a renderer. For how the pieces fit together, read the [architecture overview](docs/architecture.md).

Small changes (bug fixes, tests, documentation, a new option on an existing harness or environment) go straight
to a pull request. Changes to a public contract or to what trains, new subsystems, backends or required
dependencies, renames or removals of public names, and work that needs several pull requests or spans
repositories start with an RFC; see [When a change needs an RFC](#when-a-change-needs-an-rfc).

- [Ways to contribute](#ways-to-contribute)
- [Set up a development environment](#set-up-a-development-environment)
- [Run the tests](#run-the-tests)
- [Lint, format and type-check](#lint-format-and-type-check)
- [Commits and pull requests](#commits-and-pull-requests)
- [Add an environment](#add-an-environment)
- [Add a harness](#add-a-harness)
- [Add a renderer](#add-a-renderer)
- [Add an example](#add-an-example)
- [When a change needs an RFC](#when-a-change-needs-an-rfc)
- [Release a version](#release-a-version)

## Ways to contribute

| You want to | Start with |
|---|---|
| Report a bug | An issue with the command you ran, the config (or the smallest one that fails), the platoon commit, and the full traceback |
| Fix a bug or improve docs | A pull request; a fix comes with a test that fails without it |
| Add an environment, harness, program, reward rule, judge, sandbox provider or renderer | Usually your own package, with no change to platoon: [Extending platoon](docs/extending/index.md) has a guide to each, and the sections below sum up the common ones and cover adding one to platoon itself |
| Change a public contract or add a subsystem | An RFC ([process](docs/rfcs/README.md)) |

## Set up a development environment

**Goal:** a checkout where `platoon` runs and the test suite passes.

**Prerequisites:** git and [uv](https://docs.astral.sh/uv/). No GPU, Tinker account or model download is needed
for the tests.

1. Fork [ApGa/platoon](https://github.com/ApGa/platoon) on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-user>/platoon.git
   cd platoon
   ```

2. Python: the repository's `.python-version` names 3.12, the oldest Python platoon supports, so uv takes it for
   the checkout and downloads it if you do not have it. CI installs it first:

   ```bash
   uv python install 3.12
   ```

   CI's `test` job also runs on 3.13 and 3.14 ([Run the tests](#run-the-tests)): to use one of them, set
   `UV_PYTHON` for both `uv sync` and `uv run`.

3. Install platoon with its development dependencies, exactly as CI does:

   ```bash
   uv sync --dev --extra tinker --extra ors
   ```

   | Part | What it adds |
   |---|---|
   | `--dev` | The `dev` dependency group: pytest, pytest-asyncio, pytest-xdist (which runs the tests in parallel), ruff, pre-commit, mypy, the MCP client the tool-plane tests use, and the OpenAI SDK and litellm, which the gateway tests drive it with |
   | `--extra tinker` | The Tinker SDK (pinned) and tinker-cookbook 0.3; the gateway, the trainer and `platoon.testing` need them |
   | `--extra ors` | The openreward SDK that ORS environments are written against, number search among them |
   | `--extra hpc` (optional) | hpc-sandbox, for the `singularity` and `enroot` sandbox providers, at the release `uv.lock` names |

4. Install the git hooks: the `pre-commit` hook runs the same lint checks as CI on every commit, and the
   `commit-msg` hook checks each commit message (see [Commits and pull requests](#commits-and-pull-requests)):

   ```bash
   uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
   ```

5. Check that the command line works:

   ```bash
   uv run platoon harness list
   ```

   Expected output (with your checkout's version of platoon):

   ```text
   harness                 kind      distribution   source
   mini-swe-agent@2.4.6    spec      platoon 0.2.0  platoon/harness/builtin/specs/mini-swe-agent.toml
   openhands@1.54.0.post1  spec      platoon 0.2.0  platoon/harness/builtin/specs/openhands.toml
   platoon-bash@1          spec      platoon 0.2.0  platoon/harness/builtin/specs/platoon-bash.toml
   platoon-codeact@1       function  platoon 0.2.0  platoon.harness.builtin.codeact.harness:codeact
   platoon-noop@1          function  platoon 0.2.0  platoon.harness.builtin.oracle:noop
   platoon-oracle@1        function  platoon 0.2.0  platoon.harness.builtin.oracle:oracle
   platoon-react@1         function  platoon 0.2.0  platoon.harness.builtin.react:react
   ```

**Next:** [run the tests](#run-the-tests).

## Run the tests

CI runs core's suite in two passes, from the repository root, after `uv sync --dev --extra tinker --extra ors`. The
first spreads the test files over [pytest-xdist](https://pytest-xdist.readthedocs.io/) workers, one per CPU; the
second runs the tests marked `serial`, which cannot share the machine, in one process:

```bash
uv run pytest tests/ -v -n auto --dist loadfile -m "not serial"
uv run pytest tests/ -v -m serial
```

It runs them on each Python platoon supports, 3.12, 3.13 and 3.14: the `test` job installs its Python with
`uv python install ${{ matrix.python }}` and sets `UV_PYTHON` to it, which `uv sync` and `uv run` follow. To run the
suite on another of them, set `UV_PYTHON` for both, as in `UV_PYTHON=3.14 uv sync --dev --extra tinker --extra ors`.

Core's tests need no extension installed, and assume none is: `uv sync` at the root installs platoon alone. Each
pass ends with a summary line that counts passed and skipped tests. Skips are expected (see below); failures and
errors are not. The second pass also passes when no test is marked `serial`.

Two more CI jobs test what lives beside core. Each first-party extension (`extensions/<name>`, the distribution
`platoon-<name>`) runs its own tests in an environment of its own, with platoon from the workspace:

```bash
uv sync --package platoon-${{ matrix.extension }} --extra platoon
uv run --package platoon-${{ matrix.extension }} pytest extensions/${{ matrix.extension }}/tests -v -n auto --dist loadfile
```

where `${{ matrix.extension }}` is `textcraft`, so locally that is `uv sync --package platoon-textcraft --extra
platoon` and `uv run --package platoon-textcraft pytest extensions/textcraft/tests -n auto --dist loadfile`. The
recipes (`recipes/<name>`) check that their configs resolve and validate, with every extension installed:

```bash
uv sync --all-packages --dev --extra tinker --extra ors --extra platoon
uv run --no-sync pytest recipes/ -v
```

Run `uv sync --dev --extra tinker --extra ors` again before core's tests: it uninstalls the extensions.

| Option | What it does |
|---|---|
| `-n auto` | Starts one worker process per CPU that pytest may use. With psutil installed, pytest-xdist counts the machine's physical cores instead, also in a container or a batch job that has only some of them; give the number of workers there, such as `-n 16` |
| `--dist loadfile` | Gives each worker whole test files: a file's tests run in order in one process, as they do without `-n`, so the fixtures and module state they share keep working. The longest file bounds the run |
| `-m "not serial"`, `-m serial` | Leave the `serial` tests out of the parallel pass, and run only them in the second |

On a 32-core machine the suite takes about 8 minutes in one process and a little over 1 on 16 workers: most of its
time goes to tests that start worker processes, which import the tinker/torch stack, and wait for them.
`pytest tests/` without `-n` still runs every test, the `serial` ones too, one after another.

While you work, run only what you touch:

```bash
uv run pytest tests/harness -q                        # one directory
uv run pytest tests/test_cli.py -q -k harness         # tests whose names match
uv run pytest tests/examples -q                       # the runnable examples
uv run pytest tests/train -q -n auto --dist loadfile  # a larger part, in parallel
```

Every runnable example in `examples/<name>/` has a CPU test in `tests/examples/` that runs it against a fake
model, so the examples cannot silently break. CI runs them as part of `tests/`. Tests that drive the model gateway
render with `platoon.testing`'s toy tokenizer, which speaks the Qwen3 chat format, so they need no model download.

### Tests that skip without local files

Most of the suite is self-contained. A few groups of tests need files you may not have, and skip without them:

| Tests | Need | How to provide it |
|---|---|---|
| `tests/harness/builtin/test_codeact_golden.py`, and one test in `tests/testing/test_toy_tokenizer.py` that compares the toy tokenizer with the real one | The tokenizer of `Qwen/Qwen3-4B-Instruct-2507`: the golden fixtures hold its token ids | Download it into the Hugging Face cache (below), or set `QWEN3_4B_INSTRUCT_SNAPSHOT` to a snapshot directory elsewhere |
| `tests/renderers`, `tests/train/tinker/test_gateway_tools_routing.py` | The tokenizers and chat templates of the Qwen3.6 and Qwen3.8 snapshots named in `tests/renderers/renderer_cases.py` | The Hugging Face cache, or `PLATOON_TEST_HF_HUB` (another hub cache directory that holds them) and `QWEN36_PRESERVE_THINKING_SNAPSHOT` |
| Parts of `tests/sandbox/test_sandbox_hpc.py`, and the test of `tests/envs/test_harbor_providers.py` that checks the root filesystem layouts a run refuses against the provider | hpc-sandbox | `uv sync --dev --extra tinker --extra ors --extra hpc` |
| One test each in `tests/gateway/test_gateway_mcp.py` and `tests/envs/test_ors_concurrency.py` | The OpenHands SDK and fastmcp clients | Run them in an environment that has both, or set `PLATOON_CLIENT_VENV` to that environment's directory |

The tests look for snapshots in the standard Hugging Face cache: `HF_HUB_CACHE`, else `HF_HOME/hub`, else
`~/.cache/huggingface/hub`. To download the tokenizer files (no weights) there, at the revision the tests expect:

```bash
uv run python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554',
                        allow_patterns=['*.json', '*.txt']))"
```

The tests that load tokenizers never download anything: they set `HF_HUB_OFFLINE=1`.

### Write tests

Tests run on a CPU, without network access, a GPU or a real model. To exercise model calls, put a scripted policy
behind the real gateway. [`platoon.testing`](docs/how-to/test-extensions.md) is the public kit, which extensions
use too: `ScriptedPolicy`, `make_gateway`, `run_episode` (one task through `harness_rollout`, with a harness, an
environment and a program), `serve_policy` (for `platoon rollout` and `platoon eval` in process) and the toy
tokenizer. Prefer it in new tests. The suite also has fakes of its own:

| File | What it gives you |
|---|---|
| `tests/gateway/gateway_fakes.py` | `ScriptedSampler`: answers each model call with the next scripted completion and records the prompts |
| `tests/inline/inline_fakes.py` | `PolicySampler` (replies chosen from the last user message), `ToyService` (an environment service with a terminal `submit` tool and a read-only `look` tool), the `toy-agent` harness and `routed_task` |
| `tests/inline/conftest.py`, `tests/harness/builtin/builtin_fixtures.py` | The `make_gateway(policy)` fixture: a real `ModelGateway` with the toy tokenizer and a policy you write as a function of the prompt |
| `tests/harness/builtin/inline_runs.py` | `run_tree`: runs a function harness and its subagents on a gateway, then grades the session; `open_service_task` opens a task of an environment service |
| `tests/harness/builtin/codeact_golden.py`, `tests/harness/builtin/golden_scenarios.py` | Golden fixtures: record a scripted rollout in full and compare later runs with it; the scenarios' scaffolding and the differences they allow, with why |
| `tests/envs/scripted_trees.py` | Scripted agent trees on a live ORS session: calls, gathers and spawns, without model calls |
| `tests/harness/harness_fakes.py` | A scripted OpenAI Chat server, a process-backed sandbox and an environment service that grades files in it, for TOML spec harnesses |

Put a test next to the tests of the code it covers (`tests/envs`, `tests/harness`, `tests/gateway`,
`tests/train`, and so on). The test directories have no `__init__.py`, so pytest imports each test file by its
bare name: give every test file a name that is unique across `tests/`, `extensions/*/tests` and `recipes/`.

Tests run in parallel, next to tests in other processes and to other runs of the suite on the same machine. Keep
what a test writes in `tmp_path`, start servers on port 0, change environment variables and the working directory
with `monkeypatch`, and give anything that other processes can see a name of the test's own: the process-backed
sandbox of `tests/harness/harness_fakes.py` runs its commands as the host's processes, so an agent id that two tests
share lets one end the other's harness. Spread a slow scenario's tests over files, as the worker shutdown tests are
(`tests/train/tinker/test_worker_shutdown.py` and its siblings): one worker runs a whole file, so the longest file
bounds a parallel run. A test that cannot share the machine at all, such as one whose timing bound the load of other
processes would break, gets `@pytest.mark.serial` and runs in the second pass. When a test fails only in parallel,
run it alone (`pytest path/to/test_file.py::test_name`), then look for what it shares.

## Lint, format and type-check

`.pre-commit-config.yaml` defines four hooks. CI's `lint` job runs the two ruff hooks on every file:

```bash
uvx pre-commit run --all-files --show-diff-on-failure
```

| Hook | What it checks | Configured in |
|---|---|---|
| `ruff-check` (ruff-pre-commit `v0.14.5`, with `--fix`) | Lint rules `E` (pycodestyle errors), `F` (pyflakes) and `I` (import order); fixes what it can | `pyproject.toml`: `line-length = 120` under `[tool.ruff]`, `select = ["E", "F", "I"]` under `[tool.ruff.lint]` |
| `ruff-format` | Code formatting | `line-length` under `[tool.ruff]` |
| `ty` (`uvx ty@0.0.87 check`, at the `manual` stage) | Types, across the whole project, when you run it (below); CI does not run it | `ty.toml`: unresolved imports are ignored, and tests and the training code get looser rules |
| `conventional-pre-commit` (`v4.0.0`, at the `commit-msg` stage, with `--verbose`) | The form of each commit message, when you commit; a rejection says what the message is missing. CI does not run it | Its default types, which are the types in [Commits and pull requests](#commits-and-pull-requests) |

ty is not a required check yet: the existing code still has errors that ty reports, so the `ty` hook runs only
when you ask for it. Run it before you open a pull request, and fix what it reports in the lines you changed:

```bash
uvx pre-commit run ty --hook-stage manual --all-files
```

To run the tools one at a time:

```bash
uv run ruff check --fix path/to/file.py
uv run ruff format path/to/file.py
uvx ty@0.0.87 check
```

When a hook fixes a file during `git commit`, the commit stops; add the fixed files and commit again.

Version notes. The ruff hook runs ruff `v0.14.5`, the version that `uv.lock` installs for the `dev` group, so
`uv run ruff` and the hook agree; when you upgrade ruff in `uv.lock`, set the hook's `rev` to the same version (a
test checks that they match). The `ty` hook is pinned too, so a new ty release cannot change its results; upgrade
it in a commit of its own. Keep the `conventional-pre-commit` hook at `v4.0.0`, which this guide describes: later
releases accept any message that starts with `Merge`, and a type in any case (`Feat:`), which the pull request
title check rejects.

## Commits and pull requests

**Branches.** Branch from `main`, and keep one topic per pull request.

**Commit messages.** Commits follow [Conventional Commits](https://www.conventionalcommits.org/) in the form
`<type>: <summary>`, with a type from the table below. The summary is a short imperative sentence in sentence
case, with no final period. Conventional Commits allows a scope (`fix(envs): ...`), but platoon leaves it out.
After a blank line, a body says why:

```text
fix: Refuse spawns on an agent whose run is over

run_agent waits for the children an agent's harness left running, but a
spawn that arrived after that wait started a child that ran detached,
after the rollout had closed. run_agent now marks the agent finished once
it is done with its children, and a spawn on a finished agent is a failed
launch that runs nothing.
```

| Type | For |
|---|---|
| `feat` | A new feature, option or command |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `refactor` | A code change that neither fixes a bug nor adds a feature |
| `perf` | A performance improvement |
| `test` | Tests only |
| `build` | Packaging and dependencies |
| `ci` | The CI workflows and the git hooks |
| `chore` | Other maintenance |
| `style` | Formatting only |
| `revert` | Reverting an earlier commit |

For example:

```text
feat: Add discounting strategy for subagents
fix: Use spawn instead of fork for rollout subprocesses for better isolation
docs: Add user guides, tutorials, examples and API reference
refactor: Name the ORS task reference prefix after its format
test: Default test caches to the standard Hugging Face location
```

A change that breaks a public contract needs an RFC first (see
[When a change needs an RFC](#when-a-change-needs-an-rfc)). The commit that makes it puts `!` after the type and
adds a `BREAKING CHANGE:` footer that says what users must change:

```text
refactor!: Rename the codeact harness to platoon-codeact

Built-in harnesses are named <owner>-<name>, as platoon-react and
platoon-bash already are.

BREAKING CHANGE: configs that name the harness codeact must name it
platoon-codeact.
```

The `conventional-pre-commit` hook, installed in [step 4 of the setup](#set-up-a-development-environment),
checks the message of every `git commit` and `git merge`. It rejects a message whose subject is not one of these
types, an optional scope, an optional `!`, a colon, a space and a summary, or whose body does not follow a blank
line. It accepts a scope and does not check the summary's case or wording; reviewers check those. It accepts
`fixup!`, `squash!` and `amend!` commits, so `git rebase --autosquash` works as usual, and merge commits whose
message starts with `Merge branch`. Two of git's own messages need a fix:

- `git revert` writes `Revert "<subject>"` and commits without running the hook. Run
  `git revert --no-commit <commit>` and then `git commit`, which runs it, and write
  `revert: <the reverted commit's summary>`, keeping the `This reverts commit <hash>.` line that git puts in the
  body.
- `git merge origin/main` writes `Merge remote-tracking branch 'origin/main' into <branch>`, which the hook
  rejects, leaving the merge uncommitted. Rebase onto `main` instead, or merge a local `main`: `git merge main`
  writes `Merge branch 'main' into <branch>`.

To check a message without committing, save it to a file and run the hook on that file:

```bash
uvx pre-commit run conventional-pre-commit --hook-stage commit-msg --commit-msg-filename message.txt
```

**Pull request titles.** A pull request's title has the same form as a commit subject, for example
`fix: Close ORS sessions when a rollout is cancelled` or `docs: Add RFC 0012 on <topic>`. Maintainers
squash-merge pull requests, so the title becomes the subject of the commit on `main`. CI's `pr-title` job
(`amannn/action-semantic-pull-request@v5`) checks that the title is a Conventional Commit whose type is one of
those above, with no scope and a summary that starts with a capital letter and has no final period. When a pull
request has a single commit, the job checks that commit's subject too, because GitHub suggests it for the squashed
commit. The check runs again whenever you edit the title.

**What CI runs on a pull request.** The jobs run on GitHub's Ubuntu runners for pull requests into `main` and
for pushes to `main`; `test` runs on Python 3.12, 3.13 and 3.14, and the others use 3.12:

| Job | Runs | Fails when |
|---|---|---|
| `pr-title` | On pull requests only | The title, or the subject of a single commit, is not a Conventional Commit in the form above |
| `lint` | `uvx pre-commit run --all-files --show-diff-on-failure` | A hook fails or changes a file |
| `test` | `uv sync --dev --extra tinker --extra ors`, then the test commands above, once per Python | A test fails on any of them |

**The pull request itself.** The [pull request template](.github/pull_request_template.md) asks for each of
these:

- Say what changes and why, and how you tested it.
- Add a test for every fix and feature; a fix's test should fail without the fix.
- Update the docs, examples and command-line help that the change affects.
- Keep secrets and machine-specific paths out of code, configs and tests; read them from environment
  variables.
- Link what the change belongs to: `Fixes #123`, or `Implements RFC 0012, milestone M2` (see
  [linking an RFC to its implementation](docs/rfcs/README.md#linking-an-rfc-to-its-implementation)).

Platoon is released under the [MIT license](LICENSE), and contributions are accepted under the same license.

## Add an environment

Most environments do not need a change to platoon at all. Pick the row that fits:

| You have | Do this | Pull request to platoon |
|---|---|---|
| Tasks with tools and a grader | An ORS `Environment` class in your own package, named in a config as `service: python://your_package.module:Class` (below) | No |
| ORS environments served over HTTP | `service: ors+http://HOST:PORT/EnvName` | No |
| A classic `reset`/`step` text environment | `pt.gym.as_environment` (an ORS environment with one `act` tool, for any harness) or `pt.gym.loop` (a function harness that plays it) | No |
| Tasks with no tools, graded by the harness or program | `service: table` with `service_kwargs: {path: tasks.jsonl}` | No |
| Harbor task directories | `service: harbor` with `service_kwargs: {dataset_dir: ...}` | No |
| A small environment that is useful to everyone as a smoke test | An example (see [Add an example](#add-an-example)) or a first-party extension under `extensions/` (see [Add an extension](#add-an-extension)) | Yes |
| A new kind of environment service | An implementation of `EnvironmentService` and `EnvSession` in `platoon/envs/` | Yes, after an RFC |

`uv run platoon env list` prints the service references a config can use.

### An ORS environment in your own package

**Goal:** a gradable environment that any platoon harness can play, with a CPU test.

**Prerequisites:** the development setup above (or platoon installed with the `ors` extra) and a package of your
own that is importable in the same environment: install it with `uv pip install -e path/to/your/package`, or
put its parent directory on `PYTHONPATH`. The examples call it `my_envs`.

1. Write the environment. Its `@tool` methods are the agent's tools. A tool output with `finished=True` ends
   the episode, and its `reward` grades it:

   ```python
   # my_envs/repeat.py
   from openreward.environments import Environment, TextBlock, ToolOutput, tool
   from pydantic import BaseModel


   class Say(BaseModel):
       text: str


   class RepeatAfterMe(Environment):
       """Say the task's word back. One try per episode."""

       @classmethod
       def list_splits(cls):
           return ["train", "test"]

       @classmethod
       def list_tasks(cls, split):
           words = ["apple", "river", "stone"] if split == "train" else ["cloud"]
           return [{"id": f"{split}-{i}", "word": word} for i, word in enumerate(words)]

       def get_prompt(self):
           return [TextBlock(text=f"Call the say tool with the word {self.task_spec['word']!r}.")]

       @tool
       async def say(self, args: Say) -> ToolOutput:
           """Say one word. This ends the episode."""
           correct = args.text.strip() == self.task_spec["word"]
           reply = "Correct." if correct else "Wrong."
           return ToolOutput(blocks=[TextBlock(text=reply)], reward=float(correct), finished=True)
   ```

2. Test it through platoon's environment service, the way a rollout opens it. Platoon numbers the tasks of a
   split as `<split>:<index>`:

   ```python
   # tests/test_repeat.py
   import asyncio

   from platoon.envs import get_service


   async def say(task_id: str, word: str):
       service = get_service("python://my_envs.repeat:RepeatAfterMe")
       session = await service.open(await service.get_task(task_id), rollout_id="test")
       try:
           await session.call_tool("say", {"text": word})
           return await session.grade()
       finally:
           await session.close()


   def test_the_right_word_scores_1_and_a_wrong_one_0():
       assert asyncio.run(say("train:0", "apple")).reward == 1.0
       assert asyncio.run(say("train:0", "pear")).reward == 0.0
   ```

   ```bash
   uv run pytest tests/test_repeat.py -q
   ```

   Expected output: a line that starts with `1 passed`. To play it with a harness and a scripted policy, end to
   end as a rollout plays it, use `platoon.testing.run_episode`
   ([Test an environment, a harness or a program on CPU](docs/how-to/test-extensions.md)).

3. Add an entry for it to the `environments` list of a training or evaluation config, here `repeat.yaml`,
   with a harness that plays it:

   ```yaml
   environments:
     - name: repeat
       service: python://my_envs.repeat:RepeatAfterMe
       harness: platoon-react
   ```

4. Check the entry:

   ```bash
   uv run platoon env list repeat.yaml
   ```

   Expected output, after the table of service references:

   ```text
   Environments of repeat.yaml:
     name    weight  service                                harness        program  rollout
     repeat  tasks   python://my_envs.repeat:RepeatAfterMe  platoon-react  -        harness_rollout
   ```

**Next:** the [README quickstart](README.md#quickstart) shows a complete training config, and
`uv run platoon rollout CONFIG --task train:0` runs one task end to end once you have Tinker access.

### A new environment service in platoon

A new kind of service (another protocol, another task format) implements the protocols in
`platoon/envs/service.py` and lives in `platoon/envs/<name>/`. Write an RFC first. The contract it must keep:

- `EnvSession.grade()` never raises. An infrastructure failure is an `Outcome` with status `error` or
  `invalid` (`Outcome.infra(...)`), never a reward of 0, so it never trains.
- `close()` is idempotent, and `fork(mode)` supports the fork modes the service claims (`shared`, `copy`,
  `fresh`).
- Tools are `ToolSpec`s; mark the tool that ends the episode `terminal=True` and tools that never change state
  `read_only=True`, because agents with restricted tool access are shown tools by these flags.

Register a short name in `_BUILTIN_SERVICES` in `platoon/envs/service.py` (configs can always use the
`module:Class` path) and add it to `SERVICE_SCHEMES` in `platoon/cli.py`, so `platoon env list` shows it. Test it
in `tests/envs/`.

## Add a harness

A harness is the agent code that plays an episode. There are two kinds: a **function harness**, an async Python
function registered with `@pt.harness`, and a **TOML spec** that describes how to install and launch a
command-line agent. Both are named the same way in configs (`harness: name` or `harness: name@version`).

### A function harness in your own package

**Goal:** a harness that `platoon harness list` shows and configs can name.

**Prerequisites:** as for environments, a package that is importable in platoon's environment.

1. Write the harness. It receives an `AgentContext` and returns an `AgentResult`; `ctx.chat` samples the policy
   through the gateway and records the call for training:

   ```python
   # my_harnesses/one_shot.py
   import platoon as pt


   @pt.harness("my-one-shot")
   async def one_shot(ctx: pt.AgentContext) -> pt.AgentResult:
       """Answer the instruction with a single model call."""
       reply = await ctx.chat([{"role": "user", "content": ctx.instruction}])
       return pt.AgentResult("completed", final_message=reply.text)
   ```

   Pass `executes_model_code=True` to `@pt.harness` if the harness runs code the model wrote; it is recorded
   with every rollout. `ctx.list_tools()` and `ctx.call_tool(...)` reach the environment's tools, and
   `ctx.agents.spawn(...)` starts subagents.

2. Make platoon import the module. Either name it in each environment entry:

   ```yaml
   environments:
     - service: python://my_envs.repeat:RepeatAfterMe
       harness: my-one-shot
       package: my_harnesses.one_shot   # imported by the trainer, evaluation and every rollout worker
   ```

   or declare a `platoon.harnesses` entry point in your package's `pyproject.toml`, named after the harness, so
   the harness is available wherever the package is installed. Platoon imports it only when a config names it:

   ```toml
   [project.entry-points."platoon.harnesses"]
   my-one-shot = "my_harnesses.one_shot:one_shot"
   ```

3. Check that platoon finds it. A `module:function` reference works anywhere a harness name does:

   ```bash
   uv run platoon harness show my_harnesses.one_shot:one_shot
   ```

   Expected output:

   ```text
   my-one-shot@1  (function)
   source: my_harnesses.one_shot:one_shot
   runtime: inline
   executes_model_code: False

   Answer the instruction with a single model call.
   ```

   With the entry point installed, `uv run platoon harness list` lists `my-one-shot` too, with the distribution
   that provides it.

4. Test it on a CPU. A unit test can call the function with a stand-in context; an end-to-end test runs it on an
   environment through a real gateway with a scripted policy, with `platoon.testing.run_episode` (see
   [Test an environment, a harness or a program on CPU](docs/how-to/test-extensions.md) and
   [Write tests](#write-tests)).

### A command-line agent as a TOML spec

A spec says how to install the agent, launch it headless against platoon's OpenAI-compatible (or Anthropic)
endpoint, turn its built-in tools off where needed, and read its final message. The full format is documented at
the top of `platoon/harness/spec.py`, and `platoon/harness/builtin/specs/` has two complete specs. A minimal one:

```toml
# my-cli-agent.toml
name = "my-cli-agent"
version = "0.3.1"
dialect = "openai_chat"
timeout_s = 1800
max_steps = 50

[install]
commands = ["python3 -m pip install --user 'my-cli-agent==0.3.1'"]

[launch]
argv = ["my-cli-agent", "--model", "{model}", "--task", "{instruction}", "--max-steps", "{max_steps}"]
env = { OPENAI_BASE_URL = "{base_url}/v1", OPENAI_API_KEY = "{api_key}" }

[builtins]
tools = ["bash"]

[final_message]
file = "{state_dir}/answer.txt"
```

`{api_key}` may appear only in `[launch.env]` (and `[mcp.env]`), so the key never shows in a process listing.
Name a spec in a config by path (`harness: path/to/my-cli-agent.toml`), or ship it in a package through a
`platoon.harnesses` entry point named after the spec that loads to the spec's path. `uv run platoon harness
show path/to/my-cli-agent.toml` prints how platoon reads it.

### A built-in harness in this repository

Built-in harnesses are the ones every user gets: `platoon-codeact`, `platoon-react`, `platoon-bash` and
`mini-swe-agent`. Propose a new one in an issue first. Then:

1. Put a function harness in `platoon/harness/builtin/` and add its module to `BUILTIN_FUNCTION_MODULES` in
   `platoon/harness/registry/__init__.py`, or put a spec in `platoon/harness/builtin/specs/`. Name harnesses
   that platoon itself implements with a `platoon-` prefix; specs for third-party agents keep the agent's own
   name and version.
2. Add tests in `tests/harness/builtin/` that run it with a scripted policy, covering how it ends: a final
   answer, the step limit, a crash.
3. Say what it does in the harness function's docstring, which `platoon harness show` prints, and document
   its options in the module docstring.

## Add a renderer

A renderer turns chat messages into prompt tokens and sampled tokens back into a message. Platoon trains on the
exact tokens it sampled, so the renderer must prompt the model the way the model expects. Check whether you need
a new one at all:

| Your model | Set |
|---|---|
| One that tinker-cookbook already renders (see `tinker_cookbook.renderers.get_renderer`: `qwen3`, `qwen3_instruct`, `llama3`, `deepseekv3` and more) | `train.renderer_name` to that name |
| One whose own chat template should render the prompts, and whose tool calls are in a format platoon parses ([Models and tool calling](docs/how-to/models-and-tool-calling.md#supported-formats)) | `train.renderer_name: chat_template`, with settings such as `enable_thinking` in `train.renderer_kwargs`; the Qwen presets are `qwen3_6_preserve_thinking`, `qwen3_8_preserve_thinking` and their `_disable_thinking` variants |
| One whose chat template suits `chat_template`, but whose tool calls are in another format | `chat_template` with a tool-call parser of your own ([Write your own parser](docs/how-to/models-and-tool-calling.md#write-your-own-parser)) |
| Anything else | A new renderer (below) |

### A renderer in your own package

1. Subclass tinker-cookbook's `Renderer` and implement its three abstract methods, then register a factory
   under a name. The factory receives the tokenizer and an image processor:

   ```python
   # my_renderers/__init__.py
   from tinker_cookbook.renderers import register_renderer
   from tinker_cookbook.renderers.base import Renderer


   class MyModelRenderer(Renderer):
       """Prompts and response parsing for MyModel's chat format."""

       def __init__(self, tokenizer, image_processor=None, *, enable_thinking: bool = True):
           super().__init__(tokenizer)
           self.enable_thinking = enable_thinking  # settable with train.renderer_kwargs

       def render_message(self, message, ctx):
           """The tokens of one message: its header, its content and the end-of-turn marker."""
           raise NotImplementedError

       def get_stop_sequences(self):
           """The token ids that end a sampled turn."""
           raise NotImplementedError

       def parse_response(self, response):
           """The sampled tokens as an assistant message (with any tool calls), and whether they parsed."""
           raise NotImplementedError


   register_renderer("my_model", MyModelRenderer)
   ```

2. Register it in every process that builds a renderer. Name the module as an environment entry's `package`;
   platoon imports those modules in the trainer, in evaluation and in every rollout worker before it builds
   the renderer. Then select it with `train.renderer_name: my_model` (or `inference.renderer_name` for
   evaluation). Each key of `renderer_kwargs` sets the attribute of that name after construction, and a key
   the renderer has no attribute for is an error.

How the gateway calls a renderer: when `build_generation_prompt` takes a `tools` parameter, as
`ChatTemplateRenderer` does, it receives the OpenAI-format messages and tool definitions unchanged; otherwise
the gateway converts the messages to tinker-cookbook's format and puts the tools into the system message with
`create_conversation_prefix_with_tools`. Other keyword parameters of `build_generation_prompt` receive the
matching per-request `chat_template_kwargs`.

### A renderer in this repository

Add it under `platoon/train/tinker/renderers/` and add its factory to `RENDERER_FACTORIES` (in
`platoon/train/tinker/renderers/chat_template.py`); `register()` registers every factory in it, and platoon
calls it in each process that builds a renderer. Pin its behavior with token fixtures as `tests/renderers` does:
record the prompt token ids for a set of conversations and settings, and check that prefix merging cuts training
data only where re-rendering differs from what was sampled.

## Add an example

Runnable examples live in `examples/<name>/`: a `README.md` that states the goal, the prerequisites, the steps
and the expected output, the code, and a config. Each example has a CPU test in `tests/examples/` that runs it
against a fake model (reuse the fakes in [Write tests](#write-tests)), so the example cannot break silently.

## Add an extension

Most extensions live in packages of their own ([Publish an extension package](docs/extending/publish-a-package.md)).
A first-party extension is one this repository keeps and builds, locks and tests with core: content that platoon
maintains, such as an environment with its prompts and data. `extensions/textcraft` is the first. Lay one out as a
third party would, so that it could leave the repository unchanged:

```text
extensions/<name>/
├── pyproject.toml          the distribution platoon-<name>, a member of the repository's uv workspace
├── README.md               what it provides, how to install it, its data and its license
├── LICENSE
├── src/platoon_<name>/     the import package
│   ├── env.py              the environment: openreward and pydantic, no platoon import
│   ├── prompts.py          what imports platoon: prompt builders, named environments (entries.py), ...
│   └── data/               the data the environment reads, with SHA256SUMS
├── configs/                ready configs
└── tests/                  CPU tests, and a conftest.py
```

- **Names.** The distribution is `platoon-<name>`, the import package `platoon_<name>` and the owner of the
  components it registers `<name>` (`textcraft`, `textcraft-synth`): `platoon-*` component names are core's.
- **Dependencies.** PyPI packages only. The environment imports neither platoon nor IPython, so any ORS client can
  serve it, and its requirements are what it alone needs. What imports platoon (prompt builders, named environments,
  task generators) goes behind a `platoon` extra that requires `platoon[ors]` within one minor release
  (`platoon[ors]>=0.2,<0.3`), which platoon checks before it loads the extension's components.
  `[tool.uv.sources]` takes `platoon` from the workspace (`{ workspace = true }`) for development only.
- **Components.** Each is an entry point in the group of its kind (`platoon.environments`, ...), named as configs
  name it ([Extending platoon](docs/extending/index.md)); `platoon extensions check platoon-<name> --load` checks
  them.
- **Data.** At most 5 MB in the wheel, pinned by `data/SHA256SUMS` and a test that checks it; anything larger goes
  to the Hugging Face Hub at a pinned revision. Check that its license allows redistribution before a release.
- **Lock and tests.** `uv lock` at the repository root adds the member to `uv.lock`. Its tests run on CPU, without
  secrets, in an environment of their own (`uv sync --package platoon-<name> --extra platoon`). They may reuse
  core's test helpers in `tests/`, which their `conftest.py` puts on `sys.path`; a third party has
  `platoon.testing` instead. Core's tests never import an extension.
- **CI.** Add the name to the `extension` matrix of the `extensions` job in `.github/workflows/ci.yml`, and its
  build and checks to the release workflow.
- **Releases.** An extension has a version of its own and is published from tags of its own,
  `platoon-<name>-vX.Y.Z` ([Release the TextCraft extension](#release-the-textcraft-extension)). To publish a new
  one, give it what the release workflow has for `platoon-textcraft`: its tag pattern, the check that the tag names
  its version, an output directory and an artifact of its own, and a publish job that uploads only that artifact.
  Point its README's relative links at its tag for PyPI, as `extensions/textcraft/pyproject.toml` does, and add its
  tag pattern to the `pypi` environment and a trusted publisher for it on PyPI
  ([Set up publishing once](#set-up-publishing-once)).
- **Lint.** The extension's `pyproject.toml` has a `[tool.ruff]` of its own, with `src = ["src"]`, so its package
  sorts as first-party.

Configs that reproduce a published experiment are a recipe, not an extension: `recipes/<name>/`, with a README and a
CPU check that every config resolves ([`recipes/README.md`](recipes/README.md)).

## When a change needs an RFC

This is the full list; the README, the docs and the RFC index summarize it and link here. Write an RFC before
the code when a change does any of the following:

- **Adds or changes a public contract:** the agent API (`AgentContext`, `AgentResult`, `SubagentResult`,
  `ctx.agents.spawn`), `@pt.harness`, `@pt.program` and `EpisodeContext`, the environment service protocol, the
  harness spec format, the config schema, the `platoon` command line, the gateway's HTTP endpoints or the record
  formats.
- **Changes what trains:** credit assignment, the loss, how trajectories become training data, or which outcomes
  count as scored.
- **Adds a subsystem, a new kind of backend or a new required dependency,** such as a sandbox provider, an
  environment service backend or a training service.
- **Removes or renames public names.** Renames usually ship without aliases, so each one breaks someone's code
  or config.
- **Needs several pull requests, or changes more than one repository.**

You do not need one for bug fixes, tests, docs, behavior-preserving refactors, a new option on an existing
harness or environment, or a new environment, harness or renderer that uses the existing contracts (those can
live in your own package). If you are unsure, open an issue that describes the problem and ask.

The process (how to number an RFC, how it is reviewed, its lifecycle from `Draft` to `Accepted` and
`Implemented`, or `Rejected` or `Deferred`, and how it links to the pull requests that implement it) is in
[docs/rfcs/README.md](docs/rfcs/README.md). Start from [the template](docs/rfcs/0000-template.md).

### Changes to the public API

Two tests make every change to the public API a deliberate one:

- `tests/test_public_api.py` compares the names the public modules export (`platoon`, with `pt.gym`, and
  `platoon.testing`), with their signatures, to the snapshot in `tests/public_api.txt`. When you change the public
  API on purpose, run `python scripts/public_api.py` to rewrite the snapshot, commit it with the change, and say in
  the pull request what changed and why. Removing or renaming a public name needs an RFC first.
- `tests/test_import_boundaries.py` reads the imports in the source. Core never imports the first-party extensions
  in `extensions/` or an installed extension, neither by an import nor by an import string such as
  `python://pkg.mod:Class`, and the extensions' code and the examples import only the public names. The private
  names they still import are listed in the test, each with why: add one there only with a reason, and drop it once
  nothing imports it.

## Release a version

This section is for maintainers. Platoon is published to PyPI as `platoon` by the
[release workflow](.github/workflows/release.yml) with [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored anywhere: pushing the tag `vX.Y.Z` releases version `X.Y.Z`. The TextCraft extension is
published the same way, as `platoon-textcraft`, from tags of its own
([Release the TextCraft extension](#release-the-textcraft-extension)). A tag publishes only its own distribution, and
only a commit on `main`. The workflow also builds and checks both distributions on every pull request, without
publishing them.

### Versions

The version is set in one place, `version` in `pyproject.toml`. `platoon.__version__`, `platoon --version`, a run's
`lock.json` and the extension checks read it the same way: from `pyproject.toml` when platoon runs from a checkout, and
from the installed metadata otherwise. While the version is below 1.0, a minor release (0.3.0) may change the Python
API, the config format or the command line, and a patch release (0.2.1) only fixes bugs, so a package built on platoon
requires a range within one minor release, such as `platoon>=0.2,<0.3`. [CHANGELOG.md](CHANGELOG.md) lists what each
release changes.

### Deprecations

A form people use, such as a config key, can be deprecated before it goes: it keeps working for one minor release,
with a `pt.PlatoonDeprecationWarning` that names what replaces it and the release that removes it, and the next minor
release removes it ([RFC 0016, section 10](docs/rfcs/0016-component-configuration.md#10-deprecations)). A config
class names its deprecated keys in a class attribute, with `DeprecatedKey` from `platoon/deprecations.py`:

```python
# platoon/config_defs.py, in RolloutConfig
DEPRECATED_KEYS: ClassVar[dict[str, DeprecatedKey]] = {
    "propogate_root_success": DeprecatedKey(removed_in="0.3.0", renamed_to="propagate_root_success"),
}
```

The key keeps its field, which still works. The loader reports every config that sets it, with its path: the commands
name it once on stderr, `lock.json` records it, `platoon config validate --strict` fails on it, and the generated
config reference marks it. A Python name warns where it is used, with
`warnings.warn(message, PlatoonDeprecationWarning, stacklevel=2)`. Say in the release's notes what each deprecation
replaces, and before a minor release remove the forms it is due to remove: `grep -rn 'removed_in="0.3.0"' platoon`
lists them for 0.3.0.

### Set up publishing once

1. On GitHub, in the repository's settings (Environments), create the environment `pypi`. Limit it to the tags that
   publish, `v*` and `platoon-textcraft-v*`, and add required reviewers if someone should approve each release
   before it is uploaded. Do this before step 2: a job that names an environment that does not exist creates it
   without limits, so until then a branch's copy of the workflow could publish.
2. On PyPI, in the settings of the `platoon` project (Publishing), add a GitHub trusted publisher: owner `ApGa`,
   repository `platoon`, workflow `release.yml`, environment `pypi`. Release from that repository: the project's
   URLs and its PyPI description link to it. Add the same publisher for `platoon-textcraft` right before its first
   release, once the platoon release it requires is on PyPI
   ([Release the TextCraft extension](#release-the-textcraft-extension)). Until that release the project does not
   exist, so add it as a pending publisher on your account's Publishing page. Not earlier: PyPI creates a pending
   publisher's project the first time a publish job matches its fields, and platoon's publish job matches the same
   fields, so a platoon release in between would create the extension's project, with no files, and would fail on a
   day PyPI accepts no new projects.
3. Release only requirements that are on PyPI: PyPI refuses a release whose requirements name a URL, and an extra
   whose requirement no index has cannot be installed. hpc-sandbox, which the `hpc` extra installs, is published there
   from [its repository](https://github.com/ApGa/hpc-sandbox), and the extra takes one minor release of it, from the
   release the providers are tested against (`hpc-sandbox>=0.3.0,<0.4`). To move to another, set the range, run
   `uv lock` in the repository root (one lock covers core and the extensions), and run
   `tests/sandbox/test_sandbox_hpc.py` and `tests/envs/test_harbor_providers.py` after
   `uv sync --dev --extra tinker --extra ors --extra hpc`: their tests of the installed hpc-sandbox check the types,
   errors, options, capabilities and root filesystem layouts that the providers adapt.

4. Once the first release is on PyPI, yank `platoon 0.0.0` there, in the project's settings. It is the placeholder
   that held the name: no code, no `platoon` command, and any Python from 3.8. Until it is yanked, pip installs it
   whenever no later release can be installed, with only a warning that it lacks the extras asked for, as on Python
   3.11. pip passes over a yanked release unless a requirement pins it exactly.

### Cut a release

1. On a branch, set `version` in `pyproject.toml`, run `uv lock` in the repository root (one lock covers core and
   the extensions, and records platoon's version), and date the version's section in [CHANGELOG.md](CHANGELOG.md):
   its heading, such as `## 0.2.1 (unreleased)`, becomes `## 0.2.1 (2026-11-02)`. Where the docs mark what is new
   with "the next release" (`grep -rn "the next release" docs`), name this version instead. A minor release, such
   as 0.3.0, also moves every range that names the previous one: the extensions' and the examples' requirement on
   platoon, which would otherwise refuse the new release, and the docs and tests that show it
   (`grep -rn "0\.2, < 0\.3\|0\.2,<0\.3" --exclude=uv.lock .`). Open a pull request, such as
   `chore: Release 0.2.1`, and merge it once CI and the release workflow's checks pass.
2. Tag the merge commit and push the tag:

   ```bash
   git tag -a v0.2.1 -m "platoon 0.2.1"
   git push origin v0.2.1
   ```

3. The release workflow checks the tag, builds the sdist and the wheel, and checks them:

   ```bash
   gh api "repos/$GITHUB_REPOSITORY/compare/main...$GITHUB_SHA" --jq .status | grep -xE 'identical|behind'
   grep -q "^## ${GITHUB_REF_NAME#v} (2" CHANGELOG.md
   uv build --no-sources
   uvx twine check --strict dist/*
   uvx --from dist/*.whl platoon --version
   uvx --from dist/*.whl platoon harness list
   uvx --from dist/*.whl platoon env show platoon-number-search
   test -f "dist/platoon-${GITHUB_REF_NAME#v}-py3-none-any.whl"
   ```

   The first check fails unless the tagged commit is on `main`: GitHub's comparison of `main` with it must be
   `identical` or `behind`, not `ahead` or `diverged`. The second fails unless the changelog dates the version (step
   1), and the last when the tag does not name the version in `pyproject.toml`. Then nothing is published: delete
   the tag, fix what failed and tag again. When every check passes, the workflow uploads both files to PyPI from
   the `pypi` environment. It also builds and checks the TextCraft extension, which only its own tags publish
   ([Release the TextCraft extension](#release-the-textcraft-extension)):

   ```bash
   uv build --no-sources --package platoon-textcraft --out-dir dist-extensions
   uvx twine check --strict dist-extensions/*
   ```

   and runs the TextCraft environments from that wheel alone, without platoon:

   ```bash
   uv venv /tmp/textcraft && uv pip install --python /tmp/textcraft/bin/python dist-extensions/*.whl
   /tmp/textcraft/bin/python -I -c "import sys, platoon_textcraft.env as e; assert len(e.TextCraftSynthEnvironment.list_tasks('train-medium')) == 852; assert not [m for m in sys.modules if m.split('.')[0] in ('platoon', 'IPython')]"
   ```

4. Check the release in a fresh virtual environment (`pip install 'platoon[tinker,ors]==0.2.1'`, then
   `platoon --version` and `platoon harness list`), and create a GitHub release from the tag with the version's
   section of the changelog.

PyPI never takes the same version twice, even after a release is deleted. To fix a broken release, publish the
next patch release, and yank the broken one on PyPI.

### Release the TextCraft extension

The TextCraft extension, `platoon-textcraft`, has a version of its own, `version` in
`extensions/textcraft/pyproject.toml`, and is released from tags of its own, `platoon-textcraft-vX.Y.Z`. Its
`platoon` extra requires one minor release of platoon (`platoon[ors]>=0.2,<0.3`): release it once that release of
platoon is on PyPI.

1. On a branch, set its `version` and run `uv lock` in the repository root. Open a pull request, such as
   `chore: Release platoon-textcraft 0.1.1`, and merge it once CI and the release workflow's checks pass. When
   `version` already names the release, as it does for the first one, go to step 2.
2. Before its first release, add its publisher on PyPI ([Set up publishing once](#set-up-publishing-once), step 2).
   Tag a commit on `main` that has that version, such as the merge commit, and push the tag:

   ```bash
   git tag -a platoon-textcraft-v0.1.1 -m "platoon-textcraft 0.1.1"
   git push origin platoon-textcraft-v0.1.1
   ```

3. The release workflow builds and checks everything as in [Cut a release](#cut-a-release), step 3, except the
   changelog and platoon's version: the tagged commit must be on `main`, and the extension's wheel must have the
   version the tag names:

   ```bash
   test -f "dist-extensions/platoon_textcraft-${GITHUB_REF_NAME#platoon-textcraft-v}-py3-none-any.whl"
   ```

   When every check passes, the workflow uploads the extension's sdist and wheel, and nothing else, to PyPI from the
   `pypi` environment.
4. Check the release in a fresh virtual environment (`pip install 'platoon-textcraft[platoon]==0.1.1'`, then
   `platoon env show textcraft-synth`), and create a GitHub release from the tag.

### What the distribution holds

- **The wheel:** the `platoon` package with its prompt templates, harness specs, the testing kit's toy tokenizer
  and `py.typed`, and the license files (`LICENSE`, and the Apache-2.0 license and notice of the tool-call parsers)
  in its `.dist-info/licenses/`.
- **The sdist:** `platoon/`, `pyproject.toml`, `README.md`, `CHANGELOG.md` and the license files
  (`[tool.hatch.build.targets.sdist]` in `pyproject.toml`). Tests, docs, examples, the extensions and the recipes
  stay in the repository.
- **The project description on PyPI:** `README.md`, with its relative links and images pointed at the release's tag
  in the repository on GitHub, such as `https://github.com/ApGa/platoon/blob/vX.Y.Z/docs/README.md`
  (`[tool.hatch.metadata.hooks.fancy-pypi-readme]` in `pyproject.toml`), since PyPI cannot resolve them. They work
  once the tag is pushed, and keep showing the release's own docs after main moves on. The TextCraft extension's
  description, `extensions/textcraft/README.md`, links into its own tag the same way, such as
  `https://github.com/ApGa/platoon/blob/platoon-textcraft-vX.Y.Z/recipes/rao/README.md`.
- **The requirements:** every one comes from PyPI, since PyPI refuses direct references
  (`name @ git+https://...`), and a checkout takes them from there too: `[tool.uv.sources]` names only the
  workspace's own packages. `tests/test_packaging.py` checks both, and that every requirement is imported by platoon
  and every import is a requirement (a requirement that only raises the floor of a dependency's own, as `dill` does
  on 3.14, is named there with its reason); what only the tests or the examples import goes in the `dev` group.
- **The lower bounds:** each names the oldest release platoon works with, since pip keeps an older release that
  is already installed when it satisfies the requirement. When you add a requirement, or start using something a
  dependency added recently, check the bounds: install the lowest versions they admit into a fresh environment and
  run the tests there. This lists them:

  ```bash
  uv pip compile pyproject.toml --extra tinker --extra ors --resolution lowest-direct --only-binary :all: \
    --torch-backend cpu --python-version 3.12
  ```

  Run it with `--python-version 3.14` too: markers raise two bounds there (`datasets>=4.4` and `dill>=0.4`, whose
  picklers work on 3.14). `lowest-direct` takes the newest release of a package platoon does not require itself, so a
  release that a dependency admits but that does not work (dill 0.3 on 3.14, which datasets admits) shows only in an
  environment that already holds it: give such a package a floor of its own. Install platoon itself into that
  environment without its dependencies (`uv pip install --no-deps -e .`),
  since some tests start processes that import it. The gateway's two `litellm` tests fail with any lowest set:
  `litellm` pins `jinja2`, `tokenizers` and `click` exactly, so uv falls back to a release too old for the `openai`
  SDK.
