#!/usr/bin/env python3
"""Report validated median times from benchmark_guest.py summary.json."""
import argparse
import json
import math
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("--compiler", default="after")
    parser.add_argument("--reference", default="llvm")
    parser.add_argument("--baseline", default="before")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text())
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["variant"]] = row
    lines = ["Times are validated medians; ratios use compiler time / reference time.",
             "A ratio below 1 means the compiler is faster.", "",
             "| Case | Baseline (s) | Compiler (s) | LLVM (s) | Compiler / LLVM |",
             "| --- | ---: | ---: | ---: | ---: |"]
    ratios, improvements = [], []
    for case, variants in sorted(by_case.items()):
        def value(name):
            row = variants.get(name, {})
            return row.get("median_s") if row.get("passed") else None
        before, after, reference = map(value, [args.baseline, args.compiler, args.reference])
        ratio = after / reference if after and reference else None
        if ratio is not None:
            ratios.append(ratio)
        if before and after:
            improvements.append(before / after)
        def fmt(value):
            return f"{value:.6f}" if value is not None else "unverified"
        lines.append(f"| {case} | {fmt(before)} | {fmt(after)} | {fmt(reference)} | {fmt(ratio)} |")
    lines.append("")
    if ratios:
        geometric = math.exp(sum(map(math.log, ratios)) / len(ratios))
        lines.append(f"Compiler / LLVM geometric mean: {geometric:.4f} ({len(ratios)} validated pairs).")
        lines.append(f"Compiler faster or equal: {sum(r <= 1 for r in ratios)}/{len(ratios)}.")
    if improvements:
        geometric = math.exp(sum(map(math.log, improvements)) / len(improvements))
        lines.append(f"Baseline / compiler geometric mean speedup: {geometric:.4f}.")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
