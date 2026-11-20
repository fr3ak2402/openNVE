"""Prints the CHANGELOG.md section of a version, as release notes.

    python tools/release_notes.py v1.0.0 > release_notes.md

The section starts with "## [1.0.0]" (or "## 1.0.0") and ends at the next "## ". Without a
section, a short note with a link to the changelog is printed.
"""

import pathlib
import re
import sys


def release_notes(version: str, changelog: str) -> str:
    version = version.removeprefix("v")
    lines = changelog.splitlines()
    pattern = re.compile(rf"^##\s+\[?{re.escape(version)}\]?(\s|$)")
    for start, line in enumerate(lines):
        if pattern.match(line):
            end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            return "\n".join(lines[start + 1 : end]).strip() + "\n"
    return f"openNVE {version}. See CHANGELOG.md for the changes.\n"


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parent.parent
    print(release_notes(sys.argv[1], (root / "CHANGELOG.md").read_text(encoding="utf-8")), end="")
