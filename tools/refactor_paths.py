"""
Repo refactor helper: safe multi-file string replacements.

This utility exists because Windows PowerShell argument parsing can strip quotes when
passing multi-line Python code via `python -c`. Keeping a small helper script avoids
shell-quoting issues during repository restructures.
"""

from __future__ import annotations

import argparse
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repo root (default: .)")
    parser.add_argument("--from", dest="src", required=True, help="Old substring")
    parser.add_argument("--to", dest="dst", required=True, help="New substring")
    parser.add_argument("--glob", action="append", required=True, help="Glob pattern(s) to update")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    patterns = args.glob

    files: list[pathlib.Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))

    changed: list[pathlib.Path] = []
    for p in sorted(set(files)):
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        new = txt.replace(args.src, args.dst)
        if new != txt:
            p.write_text(new, encoding="utf-8")
            changed.append(p)

    print(f"updated files: {len(changed)}")
    for p in changed:
        print(" -", p.relative_to(root).as_posix())


if __name__ == "__main__":
    main()


