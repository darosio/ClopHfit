#!/usr/bin/env python
import argparse
import os
import re
import sys

# --- Versioned dependency patterns ---
# Matches both "Bump X from A to B" and "Update X requirement from <=A to <=B"
PAT_BUMP = re.compile(
    r"^(\s*-\s*)"  # 1: bullet + spaces
    r"(\*?\(([^)]+)\)\*?)\s*"  # 2: tag with optional asterisks; 3: inner tag
    r"(?:Bump|Update)\s+"
    r"(?:(?:pre-commit\s+hook)\s+)?"  # optional phrase
    r"(\S+?)"  # 4: package name
    r"(?:\s+(?:action|requirement))?\s+"  # optional " action" / " requirement"
    r"from\s+(\S+)\s+to\s+(\S+)"  # 5: from, 6: to
    r".*?$"  # rest (PR, etc.)
)

# --- Versionless dependency patterns (collapse to single line) ---
# e.g. "*(deps)* Update sphinx-autodoc-typehints requirement (#257)"
PAT_UPDATE_NO_VER = re.compile(
    r"^(\s*-\s*)"
    r"(\*?\(([^)]+)\)\*?)\s*"
    r"Update\s+(\S+)\s+requirement\s*\(#\d+\)\s*$"
)

# --- Repetitive non-versioned entries to collapse ---
# e.g. "*(pre-commit)* Update hooks (#255)", "*(pre-commit)* Upgrade-all"
# e.g. "*(deps)* Lock file maintenance (#756)"
PAT_COLLAPSE = re.compile(
    r"^(\s*-\s*)"
    r"(\*?\(([^)]+)\)\*?)\s*"
    r"(Update hooks|Upgrade-all|Lock file maintenance)"
    r".*$"
)


def norm_tuple(v: str) -> tuple[int, ...]:
    s = v.lstrip("vV").lstrip("<=")
    parts = re.split(r"[.\-+_]", s)
    out = []
    for p in parts:
        m = re.match(r"(\d+)", p)
        if m:
            out.append(int(m.group(1)))
        else:
            break
    return tuple(out) if out else (0,)


def condense_bumps(lines: list[str], allowed_tags: set) -> list[str]:
    """Condense repetitive dependency lines into single entries."""
    # --- Pass 1: aggregate versioned bumps ---
    agg: dict[tuple[str, str], dict] = {}
    for idx, line in enumerate(lines):
        m = PAT_BUMP.match(line)
        if not m:
            continue
        bullet, tag_text, tag, pkg, v_from, v_to = m.groups()
        if tag not in allowed_tags:
            continue
        k_from = norm_tuple(v_from)
        k_to = norm_tuple(v_to)
        key = (tag, pkg)
        if key not in agg:
            agg[key] = {
                "first_idx": idx,
                "bullet": bullet,
                "tag_text": tag_text,
                "min_from": v_from,
                "min_key": k_from,
                "max_to": v_to,
                "max_key": k_to,
            }
        else:
            if k_from < agg[key]["min_key"]:
                agg[key]["min_from"] = v_from
                agg[key]["min_key"] = k_from
            if k_to > agg[key]["max_key"]:
                agg[key]["max_to"] = v_to
                agg[key]["max_key"] = k_to

    # --- Pass 2: find first occurrence of collapsible lines ---
    collapse_first: dict[tuple[str, str], dict] = {}  # (tag, action) -> info
    versionless_first: dict[tuple[str, str], dict] = {}  # (tag, pkg) -> info
    for idx, line in enumerate(lines):
        m_c = PAT_COLLAPSE.match(line)
        if m_c:
            bullet, tag_text, tag, action = m_c.groups()
            key = (tag, action)
            if key not in collapse_first:
                collapse_first[key] = {
                    "first_idx": idx,
                    "bullet": bullet,
                    "tag_text": tag_text,
                    "action": action,
                    "count": 1,
                }
            else:
                collapse_first[key]["count"] += 1
            continue
        m_v = PAT_UPDATE_NO_VER.match(line)
        if m_v:
            bullet, tag_text, tag, pkg = m_v.groups()
            if tag not in allowed_tags:
                continue
            key = (tag, pkg)
            if key not in versionless_first:
                versionless_first[key] = {
                    "first_idx": idx,
                    "bullet": bullet,
                    "tag_text": tag_text,
                    "pkg": pkg,
                    "count": 1,
                }
            else:
                versionless_first[key]["count"] += 1

    # --- Pass 3: emit condensed output ---
    seen_bump: set[tuple[str, str]] = set()
    seen_collapse: set[tuple[str, str]] = set()
    seen_versionless: set[tuple[str, str]] = set()
    out: list[str] = []

    for idx, line in enumerate(lines):
        # Check collapsible patterns first
        m_c = PAT_COLLAPSE.match(line)
        if m_c:
            _, _, tag, action = m_c.groups()
            key = (tag, action)
            if key in seen_collapse:
                continue
            seen_collapse.add(key)
            info = collapse_first[key]
            out.append(
                f"{info['bullet']}{info['tag_text']} {info['action']}\n"
            )
            continue

        # Versionless updates
        m_v = PAT_UPDATE_NO_VER.match(line)
        if m_v:
            _, _, tag, pkg = m_v.groups()
            key = (tag, pkg)
            if tag not in allowed_tags:
                out.append(line)
                continue
            if key in seen_versionless:
                continue
            seen_versionless.add(key)
            # If also covered by a versioned bump, skip entirely
            if key in agg:
                continue
            info = versionless_first[key]
            out.append(
                f"{info['bullet']}{info['tag_text']} Update {info['pkg']}\n"
            )
            continue

        # Versioned bumps
        m = PAT_BUMP.match(line)
        if m:
            bullet, tag_text, tag, pkg, _vf, _vt = m.groups()
            if tag not in allowed_tags:
                out.append(line)
                continue
            key = (tag, pkg)
            if key in seen_bump:
                continue
            seen_bump.add(key)
            info = agg[key]
            # Drop no-op bumps (same version)
            if info["min_key"] == info["max_key"]:
                continue
            out.append(
                f"{info['bullet']}{info['tag_text']} Bump {pkg}"
                f" from {info['min_from']} to {info['max_to']}\n"
            )
            continue

        out.append(line)

    # Deduplicate identical bullet lines (e.g. repeated "- Manual cruft update")
    seen_lines: set[str] = set()
    deduped: list[str] = []
    for line in out:
        stripped = line.strip()
        if stripped.startswith("- ") and stripped in seen_lines:
            continue
        if stripped.startswith("- "):
            seen_lines.add(stripped)
        deduped.append(line)

    return deduped


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def insert_section_at_top(changelog_lines: list[str], section: list[str]) -> str:
    """
    Insert the new release section just before the first '## ' heading.
    Normalize spacing so there is exactly one blank line before and after
    the inserted section (if surrounding content exists), and ensure the
    final file ends with a single trailing newline.
    """
    # Locate insertion point
    first_release_idx = next(
        (i for i, ln in enumerate(changelog_lines) if ln.startswith("## ")), None
    )
    insert_idx = first_release_idx if first_release_idx is not None else len(changelog_lines)

    before = changelog_lines[:insert_idx]
    after = changelog_lines[insert_idx:]

    # Trim trailing blanks from 'before'
    while before and before[-1].strip() == "":
        before.pop()

    # Trim trailing blanks from 'section'
    while section and section[-1].strip() == "":
        section.pop()

    # Trim leading blanks from 'after'
    while after and after[0].strip() == "":
        after.pop(0)

    out: list[str] = []
    out.extend(before)

    if out and section:
        out.append("\n")  # exactly one blank line before section

    out.extend(section)

    if section and after:
        out.append("\n")  # exactly one blank line between section and after

    out.extend(after)

    text = "".join(out)
    # Ensure single trailing newline (no extra blank line)
    if text:
        text = text.rstrip("\n") + "\n"
    return text


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Condense bump entries from a raw release and either print to stdout "
            "(default) or insert at the top of a changelog file if --changelog is provided."
        )
    )
    ap.add_argument(
        "--raw",
        required=True,
        help="Path to RELEASE.md generated by git-cliff (should include the '## [x.y.z] - date' line).",
    )
    ap.add_argument(
        "--changelog",
        default=None,
        help="Optional path to CHANGELOG.md to update. If omitted, output is written to stdout. Use '-' or '/dev/stdout' to explicitly write to stdout.",
    )
    ap.add_argument(
        "--tags",
        default=os.environ.get("CONDENSE_TAGS", "deps,pre-commit"),
        help="Comma-separated tags to condense (default: deps,pre-commit).",
    )
    ap.add_argument(
        "--strip-header",
        type=int,
        default=2,
        help="Drop this many header lines from raw notes before condensing (default: 2).",
    )
    args = ap.parse_args()

    allowed_tags = {t.strip() for t in args.tags.split(",") if t.strip()}

    # Read and prep new notes
    raw_lines = read_lines(args.raw)
    body = raw_lines[args.strip_header :] if args.strip_header > 0 else raw_lines
    new_section = condense_bumps(body, allowed_tags)

    # Decide destination: default to stdout unless --changelog provided.
    target = args.changelog
    if target is None or target in ("-", "/dev/stdout"):
        text = "".join(new_section)
        if text:
            text = text.rstrip("\n") + "\n"  # exactly one trailing newline
        sys.stdout.write(text)
        return

    # Write only to the changelog file (no stdout)
    changelog_lines = (
        read_lines(target) if os.path.exists(target) else ["# Changelog\n", "\n"]
    )
    updated_text = insert_section_at_top(changelog_lines, new_section)
    write_text(target, updated_text)


if __name__ == "__main__":
    main()
