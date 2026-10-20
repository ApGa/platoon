# Install

**Goal:** a Python environment in which the `platoon` command runs, with the extras your runs need.

## Requirements

- **Python 3.12 or later.** Continuous integration tests 3.12, 3.13 and 3.14.
- Linux (continuous integration runs on Ubuntu). Platoon itself runs on the CPU: models are sampled and
  trained by a [Tinker-compatible training backend](first-run.md#what-a-tinker-compatible-training-backend-is).
- **git**, only to install from source.

## Install from PyPI

With pip, in a virtual environment of your own:

```bash
pip install 'platoon[tinker,ors]'
```

With [uv](https://docs.astral.sh/uv/), into the environment you have, or as a dependency of a uv project:

```bash
uv pip install 'platoon[tinker,ors]'
uv add 'platoon[tinker,ors]'
```

Pick the extras by what you run ([Extras](#extras) says what each installs):

| You run | Install |
|---|---|
| The examples, number search and other ORS environments | `platoon[tinker,ors]` |
| Harbor datasets, with Docker | `platoon[tinker]` |
| Harbor datasets on a cluster, with Singularity/Apptainer or enroot | `platoon[tinker,hpc]` |

Platoon's version is below 1.0, so a minor release (0.3.0) may change the Python API, the config format or the
command line, while a patch release (0.2.1) only fixes bugs. To stay on one minor release, require a range such as
`platoon[tinker,ors]>=0.3,<0.4`. [CHANGELOG.md](../../CHANGELOG.md) has the release notes.

Number search, the environment of [Your first run](first-run.md), ships in the package, and the TextCraft
environments are a package of their own ([The TextCraft extension](#the-textcraft-extension)). The examples, the
tutorials' files and the recipes are in the repository: to follow the tutorials,
[install from source](#install-from-source).

## Extras

| Extra | Installs | You need it for |
|---|---|---|
| `tinker` | The Tinker SDK (`tinker==0.16.1`) and `tinker-cookbook` 0.3. | `platoon train`, `platoon eval` and `platoon rollout`: anything that samples or trains a model. |
| `ors` | The OpenReward SDK (`openreward>=0.1.125`). | Environments written as OpenReward (ORS) classes: `python://` and `ors+http://` services, including number search, the TextCraft extension and `pt.gym.as_environment`. |
| `hpc` | `hpc-sandbox` 0.4 ([below](#the-hpc-extra)). | The `singularity` and `enroot` task sandboxes, for Harbor tasks on clusters without Docker. |

Install `tinker` and `ors` for everything in these guides: the trainer and the model gateway import the Tinker
SDK, and the example environments are ORS classes. A project that trains only on Harbor datasets needs no `ors`, and
`hpc` only for the cluster sandboxes.

### The hpc extra

The `hpc` extra installs [hpc-sandbox](https://pypi.org/project/hpc-sandbox/) 0.4 from PyPI, which runs the
`singularity` and `enroot` sandboxes as rootless containers (the runtimes are [system tools](#optional-system-tools)):

```bash
pip install 'platoon[tinker,ors,hpc]'
```

In a checkout, `uv sync --extra tinker --extra ors --extra hpc` installs the release that the lock file names.

To run an hpc-sandbox commit that is not released, a project managed with uv overrides the requirement in its
`pyproject.toml` (an entry in its own `[tool.uv.sources]` would apply only to its own dependencies, and hpc-sandbox
is platoon's):

```toml
[tool.uv]
override-dependencies = ["hpc-sandbox @ git+https://github.com/ApGa/hpc-sandbox.git@<commit>"]
```

With pip, name the commit next to the extra (`'hpc-sandbox @ git+https://github.com/ApGa/hpc-sandbox.git@<commit>'`);
pip takes it only when its version is one the extra accepts (0.4).

## Depend on platoon from a project

A project that runs platoon, such as a repository of configs that train on a Harbor dataset, needs no Python package
of its own: it names platoon as a dependency, with the extras its runs need and the releases it was tested with. In a
uv project, `uv add 'platoon[tinker,hpc]>=0.3,<0.4'` writes the line:

```toml
[project]
dependencies = ["platoon[tinker,hpc]>=0.3,<0.4"]
```

When only some of the project's users train, put platoon in a dependency group instead (`uv add --group train ...`),
so that the others install neither it nor the Tinker SDK and torch that its `tinker` extra brings:

```toml
[dependency-groups]
train = ["platoon[tinker,hpc]>=0.3,<0.4"]
```

`uv sync --group train` installs the group, and a plain `uv sync` uninstalls it again. Run jobs with the
environment's own `platoon` (`.venv/bin/platoon`), or with `uv run --frozen --no-sync platoon`, so that starting a job
neither changes the environment nor reaches the network, and commit `uv.lock`, so that every checkout installs the
same versions.

## Install from source

To follow the tutorials, run the examples or change platoon, clone the repository:

```bash
git clone https://github.com/ApGa/platoon.git
cd platoon
```

With uv (recommended), create the environment from the lock file:

```bash
uv sync --extra tinker --extra ors
source .venv/bin/activate          # or prefix each command with `uv run`
```

uv takes Python 3.12 for the environment, which CI tests on and the repository's `.python-version` names, and
downloads it if you do not have it.

With pip, in a virtual environment of your own:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tinker,ors]"
```

(pip does not read the lock file.)

`uv sync` makes the environment match the lock file for the extras it is given, and uninstalls everything else:
the packages of the extras you leave out, and the packages you added with `uv pip install`. So to add an extra
later, name all of yours (`uv sync --extra tinker --extra ors --extra hpc`), or add `--inexact` to keep what is
installed. `uv run` uninstalls nothing.

## The TextCraft extension

Number search ships with platoon (`platoon.envs.number_search`, named `platoon-number-search@1`): it needs only the
`ors` extra. TextCraft, the crafting environments of the recursive-agent experiments, is a first-party extension:
the distribution `platoon-textcraft` (`extensions/textcraft`), which has a version of its own. Install it with
platoon:

```bash
pip install 'platoon-textcraft[platoon]'
```

Its `platoon` extra requires `platoon[ors]` within one minor release (`>=0.3,<0.4`). In a checkout, install it from
the workspace instead:

```bash
uv sync --package platoon-textcraft --extra platoon     # the extension, platoon and the extension's test tools
uv pip install -e "extensions/textcraft[platoon]"       # or into the environment you have
```

It adds the `platoon_textcraft` package and the named environments `textcraft@1` and `textcraft-synth@1`. In a
checkout, a later `uv sync` of the root uninstalls it unless you pass `--inexact`, so install it again after one.
[`extensions/textcraft/README.md`](../../extensions/textcraft/README.md) describes the environments. If you
installed the TextCraft or number-search plugins of an earlier checkout (`plugins/`), uninstall them
(`uv pip uninstall platoon-textcraft platoon-number-search`) and install the extension again.

## Optional system tools

Some environments and harnesses run inside a task sandbox. Install the container runtime that your sandbox
provider uses:

| Provider | Needs | Typical use |
|---|---|---|
| `process` | Nothing: commands run as local subprocesses. | Tests and laptops; no isolation. |
| `docker` | The `docker` CLI (or Podman's). | Harbor tasks on a workstation or a cloud VM. |
| `singularity` | SingularityCE or Apptainer, and the `hpc` extra. | Clusters with PBS or Slurm, where containers run rootless. |
| `enroot` | enroot, and the `hpc` extra. | Clusters with enroot (and often Pyxis for Slurm). |

The site profile names the provider once for every Harbor environment run on a machine or cluster
([Write the site's profile](../how-to/clusters.md#4-write-the-sites-profile-once)); without one, Harbor
environments use `docker`. An environment entry can name its own with `service_kwargs: {provider: ...}`.

## Environment variables

| Variable | Read by | Purpose |
|---|---|---|
| `TINKER_API_KEY` | the Tinker SDK | Authenticates with your training backend; it must start with `tml-`. |
| `TINKER_BASE_URL` | the Tinker SDK | The URL of a self-hosted Tinker-compatible server, such as [miles-tinker](https://github.com/neulab/miles-tinker). |
| `TINKER_SAMPLE_TIMEOUT_SECONDS` | platoon's model gateway | Seconds one sampling call may take (default 600) when the config sets no `train.sample_timeout_seconds` or `inference.sample_timeout_seconds`; the key wins when both are set ([The sampling timeout](../how-to/train-and-eval.md#the-sampling-timeout)). |
| `WANDB_API_KEY` | Weights & Biases | Logging when `stats.wandb.mode` is not `disabled`. |
| `PLATOON_SITE` | platoon | The site profile's file, instead of `~/.config/platoon/site.yaml` ([Write the site's profile](../how-to/clusters.md#4-write-the-sites-profile-once)). |
| `OPENREWARD_API_KEY` | ORS remote environments | The key for `ors+http://` environment servers (default `local`). |
| `HF_TOKEN` | Hugging Face Hub | Downloading tokenizers of gated models, and Harbor datasets in private repositories. |
| The variable `inference.hosted.api_key_env` names (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) | `platoon eval` with a hosted model | The key of the server at `inference.hosted.base_url` ([Evaluate a hosted model](../how-to/evaluate-hosted-models.md)). |

## Check the installation

```bash
platoon --version
platoon --help
platoon harness list
```

`platoon --version` prints the installed version (`python -c "import platoon; print(platoon.__version__)"` prints
the same). `platoon --help` lists the commands (`config`, `env`, `eval`, `extensions`, `harbor`, `harness`, `prepare`,
`program`, `rollout`, `sandbox`, `tasks`, `toolchain`, `train`, `viz`). `platoon harness list` lists the built-in
harnesses, with the version of platoon that provides them:

```text
harness                 kind      distribution   source
mini-swe-agent@2.4.6    spec      platoon 0.3.0  platoon/harness/builtin/specs/mini-swe-agent.toml
openhands@1.54.0.post1  spec      platoon 0.3.0  platoon/harness/builtin/specs/openhands.toml
platoon-bash@1          spec      platoon 0.3.0  platoon/harness/builtin/specs/platoon-bash.toml
platoon-codeact@1       function  platoon 0.3.0  platoon.harness.builtin.codeact.harness:codeact
platoon-noop@1          function  platoon 0.3.0  platoon.harness.builtin.oracle:noop
platoon-oracle@1        function  platoon 0.3.0  platoon.harness.builtin.oracle:oracle
platoon-react@1         function  platoon 0.3.0  platoon.harness.builtin.react:react
```

## Install for development

To change platoon itself, install the development tools as well:

```bash
uv sync --dev --extra tinker --extra ors
uv run pre-commit install                   # lint each commit and check its message
uv run pytest tests
uvx pre-commit run --all-files              # ruff and ruff format, as CI runs them
```

These are core's tests, which need no extension installed. The TextCraft extension's tests and the recipes' checks
run in environments of their own ([Run the tests](../../CONTRIBUTING.md#run-the-tests) has the commands).

The tests need no model download: they render prompts with `platoon.testing`'s toy tokenizer. Only the tests of
real models' chat templates need local tokenizer files, and skip without them;
[CONTRIBUTING.md](../../CONTRIBUTING.md#tests-that-skip-without-local-files) lists them. Every runnable example under
`examples/` has a test in `tests/examples/` that runs it with a scripted policy.

[CONTRIBUTING.md](../../CONTRIBUTING.md) covers the rest: the checks CI runs, commit and pull request
conventions, how to add an environment, a harness or a renderer, and how to release a version. Small fixes go
straight to a pull request.
Changes to a public contract or to what trains, new subsystems, backends or required dependencies, renames or
removals of public names, and work that needs several pull requests or spans repositories start as an RFC under
[`docs/rfcs/`](../rfcs/README.md): a written proposal with the motivation, the design, the alternatives and a
rollout plan, reviewed before the code
([when a change needs an RFC](../../CONTRIBUTING.md#when-a-change-needs-an-rfc),
[the RFC process](../rfcs/README.md#how-to-propose-a-change)).

## Next steps

- [Concepts](concepts.md): the vocabulary.
- [Your first run](first-run.md): one episode, a short training run and the trajectory viewer.
- [Train on a Harbor dataset](../../README.md#train-on-a-harbor-dataset): a site profile, one config,
  `platoon prepare` and `platoon train`.
