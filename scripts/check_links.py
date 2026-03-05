#!/usr/bin/env python3
"""Simple Markdown link checker for local and remote links."""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check links in markdown files.")
    parser.add_argument("paths", nargs="*", default=["README.md", "docs"], help="Files or directories")
    parser.add_argument("--skip-remote", action="store_true", help="Skip HTTP/HTTPS validation")
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser.parse_args()


def iter_markdown_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.suffix.lower() == ".md" and path.exists():
            files.append(path)
    return files


def check_local_link(target: str, base: Path) -> bool:
    parsed = urllib.parse.urlparse(target)
    relative = parsed.path
    if not relative:
        return True
    candidate = (base.parent / relative).resolve()
    return candidate.exists()


def check_remote_link(target: str, timeout: float) -> tuple[bool, str | None]:
    headers = {"User-Agent": "posebench-link-checker/1.0"}
    request = urllib.request.Request(target, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True, None
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 405, 429, 999}:
            request = urllib.request.Request(target, method="GET", headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout):
                    return True, None
            except urllib.error.HTTPError as get_exc:
                if get_exc.code in {403, 429, 999}:
                    # Some domains block bots but are still valid links for browsers.
                    return True, None
                return False, f"HTTP {get_exc.code}"
            except Exception as get_exc:  # pragma: no cover - network behavior
                return False, str(get_exc)
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover - network behavior
        return False, str(exc)


def main() -> None:
    args = parse_args()
    files = iter_markdown_files(args.paths)
    failures: list[str] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            link = match.group(1)
            if link.startswith(("mailto:", "#")):
                continue

            if link.startswith(("http://", "https://")):
                if args.skip_remote:
                    continue
                ok, error = check_remote_link(link, timeout=args.timeout)
                if not ok:
                    failures.append(f"{file_path}: {link} ({error})")
            else:
                if not check_local_link(link, base=file_path):
                    failures.append(f"{file_path}: {link} (missing path)")

    if failures:
        print("Broken links found:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"Link check passed for {len(files)} markdown files.")


if __name__ == "__main__":
    main()
