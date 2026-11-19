"""
Converts docs/ into GitHub wiki pages.

- docs/README.md becomes Home.md, the other pages get readable names (Quick-start.md, ...)
- Links between pages are rewritten to wiki links (installation.md#x -> Installation#x)
- Links to other repository files become absolute GitHub links
- _Sidebar.md (navigation) and _Footer.md are generated

Usage:
    python tools/docs_to_wiki.py WIKI_DIRECTORY [--repository owner/name] [--branch main]

Pages in WIKI_DIRECTORY that come from docs/ are replaced, other wiki pages are kept.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# docs file -> (wiki page name, sidebar title); the order is the sidebar order
PAGES = {
    "README.md": ("Home", "Home"),
    "installation.md": ("Installation", "Installation"),
    "quickstart.md": ("Quick-start", "Quick start"),
    "configuration.md": ("Configuration", "Configuration (JSON)"),
    "api.md": ("Python-API", "Python API"),
    "time-windows.md": ("Time-windows-and-frames", "Time windows and frames"),
    "frames.md": ("Frames-and-tiles", "Frames and tiles"),
    "multiple-cameras.md": ("Multiple-cameras", "Multiple cameras"),
    "raw-files.md": ("RAW-files", "RAW files"),
    "cli.md": ("Command-line", "Command line"),
    "troubleshooting.md": ("Troubleshooting", "Troubleshooting"),
    "development.md": ("Development", "Development"),
}

GENERATED_MARKER = "<!-- generated from docs/ by tools/docs_to_wiki.py, edit docs/ instead -->"
LINK_PATTERN = re.compile(r"(\[[^\]]*\]\()([^)\s]+)(\))")


def page_name(file_name: str) -> str:
    if file_name in PAGES:
        return PAGES[file_name][0]
    # Pages that are not listed yet: "my-page.md" -> "My-page"
    stem = pathlib.Path(file_name).stem
    return stem[:1].upper() + stem[1:]


def rewrite_link(target: str, repository: str, branch: str) -> str:
    if re.match(r"^[a-z]+:", target) or target.startswith("#"):
        return target  # absolute URL, mailto:, or anchor on the same page
    path, _, anchor = target.partition("#")
    suffix = f"#{anchor}" if anchor else ""
    resolved = (DOCS / path).resolve()
    if resolved.parent == DOCS.resolve() and resolved.suffix == ".md":
        return page_name(resolved.name) + suffix
    # Other repository files and folders
    relative = resolved.relative_to(ROOT).as_posix()
    kind = "tree" if resolved.is_dir() else "blob"
    return f"https://github.com/{repository}/{kind}/{branch}/{relative}{suffix}"


def convert(markdown: str, repository: str, branch: str) -> str:
    # Do not touch links inside code blocks
    parts = re.split(r"(```.*?```)", markdown, flags=re.S)
    for index in range(0, len(parts), 2):
        parts[index] = LINK_PATTERN.sub(
            lambda match: match.group(1) + rewrite_link(match.group(2), repository, branch) + match.group(3),
            parts[index],
        )
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wiki", type=pathlib.Path, help="checked out wiki repository")
    parser.add_argument("--repository", default="fr3ak2402/openNVE")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    args.wiki.mkdir(parents=True, exist_ok=True)

    # Remove pages generated previously (renamed or deleted docs pages disappear as well)
    for path in args.wiki.glob("*.md"):
        if GENERATED_MARKER in path.read_text(encoding="utf-8", errors="ignore"):
            path.unlink()

    files = sorted(DOCS.glob("*.md"), key=lambda path: (list(PAGES).index(path.name) if path.name in PAGES else len(PAGES), path.name))
    for path in files:
        content = convert(path.read_text(encoding="utf-8"), args.repository, args.branch)
        target = args.wiki / f"{page_name(path.name)}.md"
        target.write_text(f"{GENERATED_MARKER}\n\n{content}", encoding="utf-8")
        print(f"{path.relative_to(ROOT)} -> {target.name}")

    sidebar = ["**openNVE**", ""]
    for path in files:
        title = PAGES.get(path.name, (None, page_name(path.name).replace("-", " ")))[1]
        sidebar.append(f"- [{title}]({page_name(path.name)})")
    (args.wiki / "_Sidebar.md").write_text(f"{GENERATED_MARKER}\n\n" + "\n".join(sidebar) + "\n", encoding="utf-8")
    (args.wiki / "_Footer.md").write_text(
        f"{GENERATED_MARKER}\n\nThis wiki is generated from "
        f"[docs/](https://github.com/{args.repository}/tree/{args.branch}/docs). "
        "Changes made here are overwritten, edit the files in docs/ instead.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
