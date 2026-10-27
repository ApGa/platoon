"""Packaging: what ``pip install platoon`` gets from PyPI.

The published requirements name no URL (PyPI refuses direct references) and a checkout takes them from PyPI too,
every requirement is imported by platoon itself (or raises the floor of a dependency's own, for a reason given here)
and every third-party import is a requirement, on Python 3.14 the requirements refuse the datasets and dill releases
that cannot pickle there, the classifiers name the Pythons CI tests, the version is set once (``pyproject.toml``) and
every reader gives the running platoon's, PyPI's copy of the README links into the repository at the release's tag
(and each extension's at the extension's own tag), and the install hints name the extras the way PyPI installs them.
"""

from __future__ import annotations

import ast
import importlib.metadata
import posixpath
import re
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import platoon
from platoon import extensions
from platoon.utils.config import distribution_version, run_lock

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
PROJECT = PYPROJECT["project"]
# The one lock of the workspace: core and every first-party extension (extensions/*).
LOCK = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
EXTENSIONS = sorted((REPO_ROOT / "extensions").glob("*/pyproject.toml"))
PYPI = {"registry": "https://pypi.org/simple"}
# Import names whose distribution is named otherwise (the rest lower-case to their distribution's name).
DISTRIBUTION_OF_IMPORT = {"IPython": "ipython", "yaml": "pyyaml", "tinker_cookbook": "tinker-cookbook"}
# Imports that need no requirement: (module, distribution) -> why.
UNDECLARED_IMPORTS = {
    ("platoon/testing/plugin.py", "pytest"): "a pytest plugin (pytest -p platoon.testing), which only pytest loads",
    # The openhands harness's runner runs with its toolchain's own interpreter in the sandbox (platoon toolchain
    # build), which holds the pinned SDK; platoon only copies it there and never imports it.
    ("platoon/harness/builtin/openhands/run_openhands.py", "openhands"): "runs in the OpenHands SDK toolchain",
}
# Requirements that nothing under platoon/ imports: distribution -> why platoon requires it. Each raises the floor of a
# dependency's own requirement, which pip would otherwise keep at an installed release that does not work.
UNIMPORTED_REQUIREMENTS = {
    "dill": "datasets hashes datasets with it, and on Python 3.14 a dill before 0.4 cannot pickle one",
}


def _distribution(requirement: str) -> str:
    """The normalized distribution name a requirement string names (PEP 503)."""

    name = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    assert name, f"not a requirement: {requirement!r}"
    return re.sub(r"[-_.]+", "-", name[1]).lower()


def _requirements() -> dict[str, list[str]]:
    """The published requirements: ``""`` for the core dependencies, then one entry per extra."""

    return {"": PROJECT["dependencies"], **PROJECT["optional-dependencies"]}


def _third_party_imports() -> dict[str, set[str]]:
    """Distribution name -> the modules under ``platoon/`` that import it (statically, or by a constant name passed
    to ``importlib.import_module``)."""

    imports: dict[str, set[str]] = {}
    for path in sorted((REPO_ROOT / "platoon").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names = [node.args[0].value]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in ("__future__", "platoon"):
                    continue
                distribution = DISTRIBUTION_OF_IMPORT.get(top, top.lower().replace("_", "-"))
                module = path.relative_to(REPO_ROOT).as_posix()
                if (module, distribution) not in UNDECLARED_IMPORTS:
                    imports.setdefault(distribution, set()).add(module)
    return imports


# --------------------------------------------------------------------------- requirements


def test_the_published_requirements_name_no_url():
    """PyPI refuses a distribution whose requirements are direct references (``name @ url``)."""

    urls = [line for lines in _requirements().values() for line in lines if "@" in line or "://" in line]
    assert urls == []
    assert "allow-direct-references" not in PYPROJECT.get("tool", {}).get("hatch", {}).get("metadata", {})


def test_every_requirement_is_imported_and_every_import_is_a_requirement():
    """A core dependency nothing imports belongs in the dev group (or nowhere); an import that no requirement
    covers works only while something else happens to install it."""

    imports = _third_party_imports()
    declared = {_distribution(line) for lines in _requirements().values() for line in lines}
    assert sorted(set(imports) - declared) == [], {name: sorted(imports[name]) for name in set(imports) - declared}
    for extra, lines in _requirements().items():
        unused = sorted({_distribution(line) for line in lines} - set(imports) - set(UNIMPORTED_REQUIREMENTS))
        assert unused == [], f"{extra or 'core'} requirements that nothing under platoon/ imports: {unused}"
    assert sorted(set(UNIMPORTED_REQUIREMENTS) - declared) == []
    assert sorted(set(UNIMPORTED_REQUIREMENTS) & set(imports)) == []  # an import needs no reason


def test_on_python_3_14_the_requirements_refuse_what_cannot_pickle_there():
    """pip keeps an installed release that satisfies a requirement. On Python 3.14, datasets before 4.4 and dill
    before 0.4 cannot pickle a dataset, so ``platoon train`` and ``platoon eval`` fail: there the requirements refuse
    them, and pip upgrades them. 3.12 and 3.13 keep taking both."""

    def admits(python: str, distribution: str, version: str) -> bool:
        environment = {"python_version": python, "python_full_version": f"{python}.0"}
        applying = [
            requirement
            for requirement in map(Requirement, PROJECT["dependencies"])
            if requirement.marker is None or requirement.marker.evaluate(environment)
        ]
        return all(Version(version) in each.specifier for each in applying if each.name == distribution)

    assert not admits("3.14", "datasets", "4.3.0") and admits("3.14", "datasets", "4.4.0")
    assert not admits("3.14", "dill", "0.3.9") and admits("3.14", "dill", "0.4.0")
    for python in ("3.12", "3.13"):
        assert admits(python, "datasets", "2.21.0") and admits(python, "dill", "0.3.8")


def test_a_checkout_takes_every_requirement_from_pypi():
    """A checkout installs what a release does: ``[tool.uv.sources]`` names only the workspace's own packages, at the
    root and in each extension, and every other package the lock names comes from PyPI (hpc-sandbox came from git until
    it was published there)."""

    for pyproject in [REPO_ROOT / "pyproject.toml", *EXTENSIONS]:
        sources = tomllib.loads(pyproject.read_text()).get("tool", {}).get("uv", {}).get("sources", {})
        others = {name: source for name, source in sources.items() if source != {"workspace": True}}
        assert others == {}, f"{pyproject.relative_to(REPO_ROOT)} takes these from elsewhere: {others}"
    members = set(LOCK["manifest"]["members"])
    elsewhere = {
        package["name"]: package["source"]
        for package in LOCK["package"]
        if package["name"] not in members and package["source"] != PYPI
    }
    assert elsewhere == {}


def test_the_hpc_extra_takes_the_minor_release_a_checkout_locks():
    """The hpc extra takes one minor release of hpc-sandbox, the one the lock names, from the release the providers
    are tested against: an environment that holds an older hpc-sandbox (0.5.0, whose pods fail on group-owned storage
    and under a umask such as 027; 0.4, without the pods that stacks run as; 0.2.0, which checkouts took from git)
    then upgrades instead of keeping it."""

    [line] = PROJECT["optional-dependencies"]["hpc"]
    requirement = Requirement(line)
    [locked] = [package for package in LOCK["package"] if package["name"] == "hpc-sandbox"]
    version = Version(locked["version"])
    [lowest] = [Version(spec.version) for spec in requirement.specifier if spec.operator == ">="]

    assert requirement.name == "hpc-sandbox" and not (requirement.extras or requirement.url or requirement.marker)
    assert locked["source"] == PYPI
    assert requirement.specifier == SpecifierSet(f">={lowest},<{version.major}.{version.minor + 1}")
    assert lowest.release[:2] == version.release[:2] and version in requirement.specifier
    assert not any(Version(older) in requirement.specifier for older in ("0.2.0", "0.4.0", "0.5.0"))


def test_the_lock_covers_every_extension():
    """Every first-party extension is a member of the workspace, which the one lock covers: an extension's requirements
    change with ``uv lock`` at the root, and none keeps a lock of its own."""

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    editable = {package["name"]: package["source"].get("editable") for package in lock["package"]}
    names = [tomllib.loads(pyproject.read_text())["project"]["name"] for pyproject in EXTENSIONS]
    assert names, "no extensions/*/pyproject.toml"
    for name, pyproject in zip(names, EXTENSIONS):
        assert editable.get(name) == pyproject.parent.relative_to(REPO_ROOT).as_posix(), name
        assert not (pyproject.parent / "uv.lock").exists(), f"{name} keeps a lock of its own"
    assert sorted(lock["manifest"]["members"]) == sorted(["platoon", *names])


def test_ci_tests_every_python_the_classifiers_name():
    """The classifiers name the Pythons that CI's test job runs on, from the oldest that ``requires-python``
    admits."""

    named = [
        Version(classifier.rsplit(" :: ", 1)[1])
        for classifier in PROJECT["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", classifier)
    ]
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    tested = [Version(str(python)) for python in workflow["jobs"]["test"]["strategy"]["matrix"]["python"]]
    assert named and sorted(named) == sorted(tested)
    assert SpecifierSet(PROJECT["requires-python"]) == SpecifierSet(f">={min(named)}")


# --------------------------------------------------------------------------- the distribution


def test_the_version_is_set_once_and_every_reader_gives_it():
    """``pyproject.toml`` sets the version, and ``platoon.__version__``, ``platoon --version``, ``lock.json`` and the
    extension checks read the running platoon's: here a checkout, so that file's."""

    assert "version" not in PROJECT.get("dynamic", [])
    assert platoon.__version__ == PROJECT["version"]
    assert distribution_version("platoon") == extensions.platoon_version() == PROJECT["version"]
    assert run_lock([])["versions"]["platoon"] == PROJECT["version"]


def test_a_checkouts_version_wins_over_installed_metadata(monkeypatch, capsys):
    """A checkout on the path shadows an installed platoon, whose metadata may be stale or another copy's: every
    reader gives the checkout's version, as the extension checks always did."""

    from platoon.cli import main

    real_version = importlib.metadata.version

    def stale_version(name: str) -> str:
        return "0.0.0" if name == "platoon" else real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", stale_version)
    monkeypatch.delattr(platoon, "__version__", raising=False)  # computed once, on first use
    extensions.platoon_version.cache_clear()
    try:
        assert distribution_version("platoon") == PROJECT["version"]
        assert platoon.__version__ == PROJECT["version"]
        assert run_lock([])["versions"]["platoon"] == PROJECT["version"]
        main(["--version"])
        assert capsys.readouterr().out.strip() == PROJECT["version"]
    finally:
        extensions.platoon_version.cache_clear()


def test_the_files_the_distribution_names_exist():
    assert (REPO_ROOT / "platoon/py.typed").is_file(), "type checkers read platoon's annotations only with py.typed"
    for path in PROJECT["license-files"]:
        assert (REPO_ROOT / path).is_file(), path
    for path in PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]:
        assert (REPO_ROOT / path).exists(), path


def test_pypi_s_readme_links_into_the_repository_at_the_release_tag():
    """PyPI cannot resolve README.md's relative links and images, so the build points them at the repository, at the
    tag the release workflow publishes from (``v`` and the version), so that a release's description keeps matching
    its docs after main moves on."""

    assert "hatch-fancy-pypi-readme>=24.1" in PYPROJECT["build-system"]["requires"]  # 24.1 added $HFPR_VERSION
    hook = PYPROJECT["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    [fragment] = hook["fragments"]
    text = (REPO_ROOT / fragment["path"]).read_text()
    for substitution in hook["substitutions"]:
        text = re.sub(substitution["pattern"], substitution["replacement"], text)
    text = text.replace("$HFPR_VERSION", PROJECT["version"])  # as the hook does, once the substitutions are made
    prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
    targets = re.findall(r"\]\(([^)]+)\)", prose) + re.findall(r'<img src="([^"]+)"', prose)
    relative = [target for target in targets if not re.match(r"[a-z][a-z0-9+.-]*:|#", target)]
    assert targets and relative == []
    tag = f"v{PROJECT['version']}"
    into_repository = [
        target
        for target in targets
        if re.match(r"https://(github\.com/ApGa/platoon/blob|raw\.githubusercontent\.com/ApGa/platoon)/", target)
    ]
    assert into_repository and [target for target in into_repository if f"/{tag}/" not in target] == []
    assert f"](https://github.com/ApGa/platoon/blob/{tag}/CONTRIBUTING.md#when-a-change-needs-an-rfc)" in text
    assert f'<img src="https://raw.githubusercontent.com/ApGa/platoon/{tag}/assets/' in text


@pytest.mark.parametrize("pyproject", EXTENSIONS, ids=lambda path: path.parent.name)
def test_an_extension_s_pypi_readme_links_into_the_repository_at_its_own_tag(pyproject):
    """An extension is released from tags of its own (``platoon-<name>-v`` and its version: the release workflow's
    ``platoon-textcraft-v*``), so the build points each relative link of its README at the file that link names in a
    checkout, at that tag, also when the link leaves the extension's directory (``../../recipes/...``)."""

    metadata = tomllib.loads(pyproject.read_text())
    project = metadata["project"]
    assert project.get("dynamic") == ["readme"] and "readme" not in project
    assert "hatch-fancy-pypi-readme>=24.1" in metadata["build-system"]["requires"]
    hook = metadata["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    [fragment] = hook["fragments"]
    source = (pyproject.parent / fragment["path"]).read_text()
    text = source
    for substitution in hook["substitutions"]:
        text = re.sub(substitution["pattern"], substitution["replacement"], text)
    text = text.replace("$HFPR_VERSION", project["version"])  # as the hook does, once the substitutions are made

    def prose(markdown: str) -> str:
        return re.sub(r"^```.*?^```", "", markdown, flags=re.M | re.S)

    tag = f"{project['name']}-v{project['version']}"
    directory = pyproject.parent.relative_to(REPO_ROOT).as_posix()

    def absolute(target: str) -> str:
        """Where a link of the README points: the same, or the file it names in the repository at the tag."""

        if re.match(r"[a-z][a-z0-9+.-]*:|#", target):
            return target
        path, hash_, anchor = target.partition("#")
        in_repository = posixpath.normpath(posixpath.join(directory, path))
        assert not in_repository.startswith("../") and (REPO_ROOT / in_repository).exists(), target
        slash = "/" if path.endswith("/") else ""
        return f"https://github.com/ApGa/platoon/blob/{tag}/{in_repository}{slash}{hash_}{anchor}"

    links = re.findall(r"\]\(([^)]+)\)", prose(source))
    assert re.findall(r"\]\(([^)]+)\)", prose(text)) == [absolute(target) for target in links]
    images = re.findall(r'<img src="([^"]+)"', prose(text))
    assert [image for image in images if not re.match(r"[a-z][a-z0-9+.-]*:", image)] == []


def test_a_missing_openreward_sdk_names_the_ors_extra(monkeypatch):
    from platoon.envs.ors import sdk

    monkeypatch.setitem(sys.modules, "openreward.environments", None)
    expected = (
        r"install the ors extra, e\.g\. pip install 'platoon\[ors\]', or in a platoon checkout uv sync --extra tinker "
        r"--extra ors \(uv sync removes the extras it is not given\)"
    )
    with pytest.raises(ImportError, match=expected):
        sdk.import_sdk()
