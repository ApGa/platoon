"""The Markdown docs hang together: every relative link and anchor resolves, and the docs index links every page."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
# Markdown that ships with the repository; never a whole-checkout glob, and nothing under a hidden directory or
# an installed package (an extension's own .venv, for example).
PAGES = sorted(
    {REPO / "README.md", REPO / "CONTRIBUTING.md", REPO / "CHANGELOG.md"}
    | {
        path
        for folder in ("docs", "examples", "extensions", "recipes", "platoon")
        for path in (REPO / folder).rglob("*.md")
        if not any(part.startswith(".") or part == "site-packages" for part in path.relative_to(REPO).parts)
    }
)

_FENCE = re.compile(r"^\s*(```|~~~)")
_CODE_SPAN = re.compile(r"`[^`]*`")
_LINK = re.compile(r"\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*$")
_HTML_ANCHOR = re.compile(r"<a (?:id|name)=\"(?P<anchor>[^\"]+)\"")


def _prose(path: Path) -> list[tuple[int, str]]:
    """The lines outside fenced code blocks, with their line numbers."""

    lines, fenced = [], False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            lines.append((number, line))
    return lines


def _anchors(path: Path) -> set[str]:
    """GitHub's anchors for a page: one per heading (repeats numbered), plus explicit ``<a id=...>`` anchors."""

    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for _, line in _prose(path):
        anchors.update(match["anchor"] for match in _HTML_ANCHOR.finditer(line))
        heading = _HEADING.match(line)
        if not heading:
            continue
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading["title"]).replace("`", "")
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        anchors.add(f"{slug}-{seen[slug]}" if slug in seen else slug)
        seen[slug] = seen.get(slug, 0) + 1
    return anchors


def _links(path: Path) -> list[tuple[int, str]]:
    return [
        (number, match["target"])
        for number, line in _prose(path)
        for match in _LINK.finditer(_CODE_SPAN.sub("", line))
        if not re.match(r"^[a-z][a-z0-9+.-]*:", match["target"])  # http:, https:, mailto: and the like
    ]


@pytest.mark.parametrize("page", PAGES, ids=lambda path: str(path.relative_to(REPO)))
def test_relative_links_and_anchors_resolve(page: Path) -> None:
    broken = []
    for number, target in _links(page):
        path_text, _, anchor = target.partition("#")
        path = (page.parent / path_text).resolve() if path_text else page
        if not path.exists():
            broken.append(f"line {number}: {target} (no such file)")
        elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
            broken.append(f"line {number}: {target} (no such heading)")
    assert not broken, f"{page.relative_to(REPO)} has broken links:\n" + "\n".join(broken)


@pytest.mark.parametrize("index", ["README.md", "index.md"])
def test_the_docs_index_links_every_page(index: str) -> None:
    linked = {(DOCS / path).resolve() for _, target in _links(DOCS / index) if (path := target.partition("#")[0])}
    # Single RFCs are listed by the RFC index, which tests/test_rfc_index.py checks.
    pages = {path for path in DOCS.rglob("*.md") if path.parent != DOCS / "rfcs" or path.name == "README.md"}
    missing = sorted(str(path.relative_to(DOCS)) for path in pages - linked - {DOCS / "README.md", DOCS / "index.md"})
    assert not missing, f"docs/{index} does not link {missing}"


# The sections that tell readers how to contribute, and the page each one is on.
CONTRIBUTING_SECTIONS = [
    ("README.md", "Contributing and development"),
    ("CONTRIBUTING.md", "When a change needs an RFC"),
    ("docs/README.md", "Contributing"),
    ("docs/index.md", "Contributing"),
    ("docs/getting-started/install.md", "Install for development"),
]


def _section(path: Path, title: str) -> list[str]:
    """The prose lines under the heading ``title``, up to the next heading of the same or a higher level."""

    lines, level = [], None
    for _, line in _prose(path):
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading and level is not None and len(heading[1]) <= level:
            break
        if level is not None:
            lines.append(line)
        elif heading and heading[2] == title:
            level = len(heading[1])
    assert level is not None, f"{path.relative_to(REPO)} has no section {title!r}"
    return lines


@pytest.mark.parametrize(("page", "title"), CONTRIBUTING_SECTIONS)
def test_contributing_sections_point_at_the_rfc_process(page: str, title: str) -> None:
    """Every contributing or development section links the RFC process and the list of changes that need an RFC."""

    text = "\n".join(_section(REPO / page, title))
    assert "rfcs/README.md" in text, f"{page} '{title}' does not link the RFC process (docs/rfcs/README.md)"
    if page != "CONTRIBUTING.md":
        assert "#when-a-change-needs-an-rfc" in text, f"{page} '{title}' does not link when a change needs an RFC"
