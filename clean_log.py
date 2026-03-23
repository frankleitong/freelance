#!/usr/bin/env python3
"""Clean repetitive noise lines from RTF log files.

Usage: python3 clean_log.py [input.rtf] [output.rtf]
Defaults: input=Untitled.rtf, output=Untitled_cleaned.rtf
"""

import re
import sys
import os

# Noise patterns: (category_name, regex, show_all_deleted)
# show_all_deleted=True means print every unique deleted line (used for errors)
NOISE_PATTERNS = [
    ("DRVSTUB_LOG sendStarsSQE", r"\[DRVSTUB_LOG\].*sendStarsSQE", False),
    ("WARN scalar overflow",     r"\[WARN\].*check_status overflow", False),
    ("info block_start",         r"\[info\].*\[block_start\]", False),
    ("INFO IDEDD/HDC",           r"\[INFO\] IDEDD", False),
    ("info block_end",           r"\[info\].*\[block_end\]", False),
    ("error vec_err_idata_inf_nan", r"\[error\].*vec_err_idata_inf_nan", True),
    ("info TASK_DONE",           r"\[info\].*\[TASK_DONE\]", False),
    ("DRVSTUB_LOG sq_addr",      r"\[DRVSTUB_LOG\].*sendStarsSQE:sq_addr", False),
]

# Compile patterns
COMPILED_PATTERNS = [(name, re.compile(pat), show_all) for name, pat, show_all in NOISE_PATTERNS]


def classify_line(line):
    """Return the noise category name if the line matches, else None."""
    for name, pattern, _ in COMPILED_PATTERNS:
        if pattern.search(line):
            return name
    return None


def clean_log(input_path, output_path):
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    kept_lines = []
    # category -> {"count": int, "sample": str, "all_lines": [str], "show_all": bool}
    deleted = {}

    for line in lines:
        category = classify_line(line)
        if category is None:
            kept_lines.append(line)
        else:
            if category not in deleted:
                show_all = next(s for n, _, s in COMPILED_PATTERNS if n == category)
                deleted[category] = {
                    "count": 0,
                    "sample": line.rstrip(),
                    "all_lines": [],
                    "show_all": show_all,
                }
            deleted[category]["count"] += 1
            deleted[category]["all_lines"].append(line.rstrip())

    # Write cleaned file
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    # Print summary
    total_deleted = sum(v["count"] for v in deleted.values())
    total_kept = len(kept_lines)

    print("=" * 70)
    print("LOG CLEANING SUMMARY")
    print("=" * 70)
    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Total lines: {len(lines)}  |  Deleted: {total_deleted}  |  Kept: {total_kept}")
    print("=" * 70)

    for category, info in deleted.items():
        print(f"\n--- {category} ({info['count']} lines deleted) ---")
        if info["show_all"]:
            print("  ALL deleted lines:")
            # Deduplicate by stripping RTF escapes for display
            seen = set()
            for l in info["all_lines"]:
                # Simplify for display: strip RTF escape sequences
                display = re.sub(r"\\$", "", l)  # trailing backslash
                display = re.sub(r"\\'[0-9a-fA-F]{2}", " ", display)  # \'xx
                display = display.strip()
                if display not in seen:
                    seen.add(display)
                    print(f"    {display}")
            if len(seen) < info["count"]:
                print(f"  ({info['count'] - len(seen)} duplicate lines omitted)")
        else:
            sample = info["sample"]
            sample = re.sub(r"\\$", "", sample)
            sample = re.sub(r"\\'[0-9a-fA-F]{2}", " ", sample)
            print(f"  Sample: {sample.strip()}")

    print("\n" + "=" * 70)
    print("KEPT LINES (non-RTF content):")
    print("=" * 70)
    for line in kept_lines:
        stripped = line.strip()
        # Skip pure RTF markup for display
        if stripped and not stripped.startswith("{\\") and not stripped.startswith("\\") and stripped != "}":
            display = re.sub(r"\\'[0-9a-fA-F]{2}", " ", stripped)
            display = re.sub(r"\\$", "", display)
            if display.strip():
                print(f"  {display.strip()}")

    print("=" * 70)
    print("Done. Please review the summary above.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "Untitled.rtf")
    output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, "Untitled_cleaned.rtf")
    clean_log(input_file, output_file)
