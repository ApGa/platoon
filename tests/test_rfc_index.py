"""The RFC index (``docs/rfcs/README.md``) lists every RFC with the title and status the RFC gives itself, and
the template has the sections the index's format list names."""

from __future__ import annotations

import re
from pathlib import Path

RFC_DIR = Path(__file__).resolve().parents[1] / "docs" / "rfcs"
INDEX = RFC_DIR / "README.md"
TEMPLATE = RFC_DIR / "0000-template.md"
STATUSES = ("Draft", "Accepted", "Implemented", "Rejected", "Deferred")

_ROW = re.compile(r"^\| \[(?P<number>\d{4})\]\((?P<link>[^)]+)\) \| (?P<title>.+?) \| (?P<status>[^|]+?) \|$", re.M)
_FILE_NAME = re.compile(r"^\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")


def rfc_files() -> dict[str, Path]:
    """Every RFC by number, the template excluded."""

    files = [path for path in sorted(RFC_DIR.glob("*.md")) if path.name not in {"README.md", TEMPLATE.name}]
    numbers = [path.name[:4] for path in files]
    assert len(numbers) == len(set(numbers)), f"two RFCs share a number: {numbers}"
    return dict(zip(numbers, files))


def index_rows() -> dict[str, re.Match[str]]:
    rows: dict[str, re.Match[str]] = {}
    for row in _ROW.finditer(INDEX.read_text()):
        assert row["number"] not in rows, f"RFC {row['number']} is listed twice"
        rows[row["number"]] = row
    return rows


def title_and_status(path: Path) -> tuple[str, str]:
    """The title from ``# RFC NNNN: Title`` and the status from the status table, or from a ``Status:`` line."""

    text = path.read_text()
    title = re.fullmatch(rf"# RFC {path.name[:4]}: (.+)", text.splitlines()[0])
    assert title, f"{path.name} must start with '# RFC {path.name[:4]}: <title>'"
    status = re.search(r"^\| Status \| (.+?) \|$", text, re.M) or re.search(r"^Status: (.+?)(?:\.\s|\.?$)", text, re.M)
    assert status, f"{path.name} gives no status"
    return title[1], status[1]


def test_rfc_file_names_are_numbered_and_hyphenated():
    for path in [TEMPLATE, *rfc_files().values()]:
        assert _FILE_NAME.match(path.name), f"{path.name} is not NNNN-lower-case-words.md"


def test_the_index_lists_every_rfc_with_its_title_and_status():
    files, rows = rfc_files(), index_rows()
    assert sorted(rows) == sorted(files), "the index and docs/rfcs/ list different RFCs"
    for number, path in files.items():
        row = rows[number]
        title, status = title_and_status(path)
        assert row["link"] == path.name
        assert row["title"] == title, f"RFC {number}: the index says {row['title']!r}, the RFC {title!r}"
        assert row["status"] in STATUSES, f"RFC {number}: {row['status']!r} is not one of {STATUSES}"
        claimed = row["status"]
        assert claimed.lower() in status.lower(), f"RFC {number}: the index says {claimed}, the RFC says {status}"


def test_the_index_explains_every_status():
    text = INDEX.read_text()
    for status in STATUSES:
        assert f"| `{status}` |" in text, f"the lifecycle table does not explain {status}"


def test_the_template_has_the_sections_the_index_lists():
    index = INDEX.read_text()
    format_section = index[index.index("## Format") :]
    listed = re.findall(r"^\d+\. \*\*(.+?)\*\*", format_section, re.M)
    template = TEMPLATE.read_text()
    headings = re.findall(r"^## (.+)$", template, re.M)
    required = [heading for heading in headings if heading != "The hard parts"]
    assert required == listed, f"the template's sections {required} differ from the index's format list {listed}"
    assert template.startswith("# RFC NNNN: ")
    assert re.search(r"^\| Status \| Draft \|$", template, re.M)
    rows = re.findall(r"^\| (\w+(?: \w+)?) \| ", template, re.M)
    for row in ("Status", "Created", "Authors", "Builds on", "Discussion", "Implementation"):
        assert row in rows, f"the template's status table has no {row} row"
        assert f"`{row}`" in format_section, f"the index's format section does not name the {row} row"
