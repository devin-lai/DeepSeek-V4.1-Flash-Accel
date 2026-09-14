#!/usr/bin/env python3
"""Check staged public files without importing the GPU runtime (Python 3.11+)."""
from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "possible credential": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,})\b"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "personal home path": re.compile(r"/(?:Users|home)/[\w.-]+/|/root/(?!\.cache/)[\w.-]+/"),
}
# Only these visually reviewed illustrations may bypass the text scan. Updating
# an image requires inspecting it and updating its exact content digest here.
REVIEWED_IMAGES: dict[str, str] = {
    "docs/assets/cuda-graphs.webp": "8c4803ad90200a299d0fe9a437ef052add3bb615f06b3f5c4fd0209161c4f366",
    "docs/assets/expert-placement.webp": "835b269750ced3b75dfdc62b3f333ade5ad005ba66a65b825d3920a07d61bd61"
}
LINK = re.compile(r"\[[^\]\n]+\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def git(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, **kwargs)


def without_fences(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            lines.append("")
        else:
            lines.append(line if fence is None else "")
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    found = set()
    counts: Counter = Counter()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", without_fences(text), re.M):
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        found.add(slug + suffix)
        counts[slug] += 1
    found.update(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    return found


def main() -> int:
    result = git("ls-files", "--cached", "-z", check=True)
    paths = result.stdout.decode().strip("\0").split("\0")
    errors = []
    if not paths or paths == [""]:
        print("No staged public files found.", file=sys.stderr)
        return 1
    if git("diff", "--quiet", "--", ".gitignore").returncode:
        errors.append(".gitignore: stage ignore-rule changes before running this check")
    ignored = git("check-ignore", "--no-index", "-z", "--stdin",
                  input=("\0".join(paths) + "\0").encode())
    if ignored.returncode not in (0, 1):
        errors.append("Could not check ignore rules")
    for path in ignored.stdout.decode().split("\0"):
        if path:
            errors.append(f"{path}: staged file matches an ignore rule")

    contents = {}
    for path in paths:
        data = git("show", f":{path}", check=True).stdout
        if path in REVIEWED_IMAGES:
            if hashlib.sha256(data).hexdigest() != REVIEWED_IMAGES[path]:
                errors.append(f"{path}: illustration changed; review it and update its digest")
            if not (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
                errors.append(f"{path}: expected a WebP illustration")
            if len(data) > 2 * 1024 * 1024:
                errors.append(f"{path}: illustration exceeds the 2 MiB page-asset limit")
            continue
        try:
            contents[path] = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{path}: binary artifact needs explicit publication review")
            continue
        for number, line in enumerate(contents[path].splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    # Never print matched credentials or private data.
                    errors.append(f"{path}:{number}: {label}")
        suffix = Path(path).suffix
        try:
            if suffix == ".py":
                ast.parse(contents[path], filename=path)
            elif suffix == ".json":
                json.loads(contents[path])
            elif suffix == ".toml":
                tomllib.loads(contents[path])
            elif suffix in {".sh", ".env"} or path.endswith(".env.example"):
                shell = subprocess.run(["bash", "-n"], input=data, capture_output=True)
                if shell.returncode:
                    errors.append(f"{path}: invalid shell syntax")
        except (SyntaxError, ValueError) as exc:
            # Parsers may include source text in exceptions; only report the type.
            errors.append(f"{path}: invalid syntax ({type(exc).__name__})")

    published = set(paths)
    heading_ids = {p: anchors(t) for p, t in contents.items() if p.endswith(".md")}
    for path, text in contents.items():
        if not path.endswith(".md"):
            continue
        prose = without_fences(text)
        for match in LINK.finditer(prose):
            url = urlsplit(match[1].strip("<>"))
            if url.scheme or url.netloc:
                continue
            dest = unquote(url.path)
            target = posixpath.normpath(posixpath.join(posixpath.dirname(path), dest)) if dest else path
            number = prose[:match.start()].count("\n") + 1
            # A local file can exist while being ignored and absent from the publication.
            is_dir = any(p.startswith(target.rstrip("/") + "/") for p in published)
            if target not in published and not is_dir:
                errors.append(f"{path}:{number}: link target is not published: {target}")
            elif url.fragment and target in heading_ids and unquote(url.fragment) not in heading_ids[target]:
                errors.append(f"{path}:{number}: missing heading in {target}: #{url.fragment}")

    if errors:
        print("Public repository check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Public repository check passed: {len(paths)} staged files; ignore rules, common credential patterns, home paths, syntax, and local Markdown links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
