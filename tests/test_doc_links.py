"""Documentation link integrity test (Priority 1.3).

Parses every tracked Markdown file in the repository and fails if:
1. Any `file:///` link (laptop-only absolute URI) appears in a Markdown link.
2. Any relative Markdown link points to a target file or directory that does not exist.
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


def _iter_markdown_files() -> list[Path]:
    files = []
    for p in REPO_ROOT.rglob("*.md"):
        rel_parts = p.relative_to(REPO_ROOT).parts
        if any(part in {".git", ".venv", "venv", "node_modules", "helper_bench"} for part in rel_parts):
            continue
        files.append(p)
    return sorted(files)


def test_no_file_uri_or_broken_relative_links_in_markdown() -> None:
    file_uri_errors: list[str] = []
    broken_links: list[str] = []

    for md_path in _iter_markdown_files():
        text = md_path.read_text(encoding="utf-8")
        # Strip fenced code blocks so code examples are not parsed as links
        stripped = FENCED_CODE_RE.sub("", text)
        rel_md = md_path.relative_to(REPO_ROOT).as_posix()

        for match in MD_LINK_RE.finditer(stripped):
            target = match.group(1).strip()
            if not target or target.startswith("#"):
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("file:///"):
                file_uri_errors.append(f"{rel_md} -> {target}")
                continue

            # Strip optional #anchor or ?query
            clean_target = target.split("#", 1)[0].split("?", 1)[0]
            if not clean_target:
                continue
            resolved = (md_path.parent / clean_target).resolve()
            if not resolved.exists():
                broken_links.append(f"{rel_md} -> {target}")

    assert not file_uri_errors, (
        "Found laptop-only file:/// links in Markdown:\n" + "\n".join(file_uri_errors)
    )
    assert not broken_links, (
        "Found broken relative links in Markdown:\n" + "\n".join(broken_links)
    )
