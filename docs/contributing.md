# Contributing

## Development setup

```bash
git clone https://github.com/ApGa/platoon.git
cd platoon
uv sync --dev
```

`uv sync --dev` installs the core package plus the dev tools — `pytest`, `pytest-asyncio`, `ruff`,
`mypy` and `pre-commit` — without pulling in a training backend. That is deliberate: the core,
the tests, and the docs must all stay installable on a laptop with no CUDA.

Add a backend only when you need one:

```bash
uv sync --extra areal      # or --extra tinker; the two conflict by design
```

## Tests

```bash
uv run pytest tests/ -v                       # the whole suite, as CI runs it
uv run pytest tests/test_registry_components.py -v
uv run pytest tests/ -k subagent -v           # one theme
uv run pytest tests/ -x -q                    # stop at the first failure
```

The suite is unusually load-bearing for a research codebase — many tests pin *contracts* rather
than implementations, so a failing test usually means a behaviour change rather than a flaky
environment. If you are unsure what a component is allowed to do, its test is the best available
specification.

## Lint, format and types

The repository uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting, and
[`ty`](https://docs.astral.sh/ty/) for type checking. All three run through `pre-commit`, which is
also what CI runs:

```bash
uvx pre-commit install                 # once, to enable the git hooks
uvx pre-commit install --hook-type commit-msg
uvx pre-commit run --all-files         # what CI executes
```

Individually:

```bash
uv run ruff check --fix .
uv run ruff format .
uvx ty check
```

Ruff is configured with `line-length = 120` and the `E`, `F` and `I` rule sets. The three
`textual` TUI modules are excluded from `ty` — they lean heavily on dynamic attributes.

## Commit messages

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/); a
`conventional-pre-commit` hook enforces it locally and a
[semantic PR title check](https://github.com/amannn/action-semantic-pull-request) enforces it on
pull requests. Pull request *titles* are checked too, so `fix: guard against empty rollout groups`
passes and `fixed a bug` does not.

## Working on the documentation

The docs are a MkDocs Material site under `docs/`, with dependencies deliberately separated from
the project's own lockfile so the site can build without CUDA or a backend:

```bash
uv venv .docs-venv --python 3.12
uv pip install --python .docs-venv -r docs/requirements.txt
./.docs-venv/bin/mkdocs serve
```

That serves the site at `http://127.0.0.1:8000/platoon/` with live reload.

Before opening a pull request, build the way CI does — `--strict` promotes broken internal links
and other warnings to errors:

```bash
./.docs-venv/bin/mkdocs build --strict
```

### Conventions for docs changes

- **Code excerpts are copied, never paraphrased.** If an example does not exist in the repository,
  say so; if it does, quote it and give the path.
- **Every configuration key comes from a dataclass.** Defaults in the reference tables are read out
  of the source, not remembered. When you change a default, update
  `docs/reference/configuration.md` in the same pull request.
- **New pages go in the `nav` in `mkdocs.yml`.** A page that is not in the nav is unreachable and
  `--strict` will not catch it.
- Prose is wrapped at roughly 100 characters. Diagrams are Mermaid, not images, so they stay
  editable and theme-aware.

## Adding a plugin to the repository

Plugins live in `plugins/<name>/` and are separate `uv` projects that depend on the root package
through a path source. [Packaging a plugin](customization/packaging.md) covers the layout, the
namespace-package convention, and the entry point that makes your components discoverable. A new
plugin should also add a row to the [plugin catalog](reference/plugins.md) and ship a README with
its own setup requirements.

## Reporting problems

Issues and feature requests go to
[github.com/ApGa/platoon/issues](https://github.com/ApGa/platoon/issues). For a training bug,
include the config (redacting paths and keys), the backend, the node count, and — if you have one —
the rollout event log, which is usually more informative than the traceback.
