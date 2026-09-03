# Contributing

Contributions are welcome but not required to build on Platoon: a plugin is an ordinary Python
package, so a task plugin or a capability plugin can live in your own repository, with no fork and
nothing to upstream. See [Packaging a plugin](plugins/index.md). Plugins that do ship here live in
`plugins/<name>/` and add a row to the [plugin catalog](plugins/catalog.md).

## Development setup

```bash
git clone https://github.com/ApGa/platoon.git
cd platoon
uv sync --dev
uv sync --extra areal      # optional; or --extra tinker, the two conflict by design
```

`uv sync --dev` installs the core package plus `pytest`, `pytest-asyncio`, `ruff`, `mypy` and
`pre-commit`, without a training backend, so the core, the tests and the docs stay installable on
a laptop with no CUDA. Add a backend extra only when you need one.

## Tests

```bash
uv run pytest tests/ -v                       # the whole suite, as CI runs it
uv run pytest tests/ -k subagent -v           # one theme
uv run pytest tests/ -x -q                    # stop at the first failure
```

Many tests pin contracts rather than implementations, so a component's test is the clearest
statement of what it is allowed to do.

## Lint, format and types

Platoon uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting and
[`ty`](https://docs.astral.sh/ty/) for type checking. Both run through `pre-commit`, which is what
CI runs too:

```bash
uvx pre-commit install                 # once, to enable the git hooks
uvx pre-commit install --hook-type commit-msg
uvx pre-commit run --all-files

uv run ruff check --fix .              # or run the tools individually
uv run ruff format .
uvx ty check
```

Ruff is configured with `line-length = 120` and the `E`, `F` and `I` rule sets.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/), enforced by a
`conventional-pre-commit` hook and, for pull request titles, by a
[semantic PR title check](https://github.com/amannn/action-semantic-pull-request). So
`fix: guard against empty rollout groups` passes and `fixed a bug` does not.

## Building the documentation

The site is MkDocs Material under `docs/`, with its own dependency file so it builds without a
backend:

```bash
uv venv .docs-venv --python 3.12
uv pip install --python .docs-venv -r docs/requirements.txt
./.docs-venv/bin/mkdocs serve            # live reload at http://127.0.0.1:8000/platoon/
./.docs-venv/bin/mkdocs build --strict   # what CI runs; warnings become errors
```

Run the `--strict` build before opening a pull request: it catches broken internal links. When you
change the docs:

- Copy code excerpts from the repository instead of paraphrasing them, and give the file path.
- Read configuration keys and defaults from the dataclass that defines them, and update
  [Configuration](reference/configuration.md) in the same pull request as a change.
- Add new pages to the `nav` in `mkdocs.yml`; a page outside the nav is unreachable.
- Wrap prose at roughly 100 characters, and draw diagrams in Mermaid so they stay theme-aware.

## Engineering walkthroughs

Line-by-line traces through the internals — a training run end to end, a rollout group being
scored and filtered, a trajectory tree becoming a batch, a delegation call, the anatomy of a
plugin — live in the repository under `notes/walkthroughs/`. They are written for people modifying
the framework itself; this site is the reference for using it.

## Reporting problems

Issues and feature requests go to [the issue tracker](https://github.com/ApGa/platoon/issues). For
a training bug, include the config (redacting paths and keys), the backend, the node count and, if
you have one, the rollout event log.
