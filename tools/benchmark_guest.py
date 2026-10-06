#!/usr/bin/env python3
"""Run a prepared benchmark directory, checking every result before timing it.

The directory contains cases/*.in, cases/*.out and <variant>/*.elf.
Executions are sequential and variant order alternates between repetitions.
No VM is started or stopped by this program.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import tarfile
import time


def normalize(data):
    return b" ".join(data.split())


def unpack_sparse(archive, root):
    # Some SysY global initializers produce hundreds of MB of zero-filled
    # .data. Preserve the bytes while leaving zero chunks as filesystem holes.
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            target = (root / member.name).resolve()
            if target == root or root not in target.parents:
                raise ValueError("archive member escapes benchmark directory")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("only regular benchmark files are supported")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as dest:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    if chunk.count(0) == len(chunk):
                        dest.seek(len(chunk), 1)
                    else:
                        dest.write(chunk)
                dest.truncate(member.size)
            target.chmod(member.mode & 0o777)


def run_one(executable, input_path, expected, timeout, runner):
    start = time.monotonic()
    try:
        with input_path.open("rb") if input_path is not None and input_path.exists() else open(os.devnull, "rb") as inp:
            result = subprocess.run([*runner, str(executable)], stdin=inp,
                                    capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "wall_s": timeout}
    elapsed = time.monotonic() - start
    # Runtime timing goes to stderr. Keep it separate from stdout so it cannot
    # split a final integer printed without a newline.
    actual = result.stdout + str(result.returncode).encode()
    timers = re.findall(rb"TOTAL:\s*(\d+)H-(\d+)M-(\d+)S-(\d+)us", result.stderr)
    seconds = sum(((int(h) * 60 + int(m)) * 60 + int(s)) + int(us) / 1e6
                  for h, m, s, us in timers)
    matches = (hashlib.sha256(normalize(actual)).hexdigest() == expected
               if isinstance(expected, str) else normalize(actual) == normalize(expected))
    return {"status": "PASS" if matches else "FAIL",
            "returncode": result.returncode, "wall_s": elapsed,
            "time_s": seconds if timers else elapsed,
            "metric": "sysy-total" if timers else "wall",
            "stderr": result.stderr.decode(errors="replace")[-1000:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--unpack", type=pathlib.Path, help="Extract a payload as sparse files")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--qemu-user", help="Optional qemu-riscv64 executable")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    if args.runs < 1 or args.timeout <= 0:
        parser.error("runs and timeout must be positive")
    root = args.root.resolve()
    if args.unpack:
        unpack_sparse(args.unpack, root)
        return 0
    if not args.variants:
        parser.error("supply at least one variant")
    # Keep all variants on the same virtual CPU, sharing its TCG code cache.
    # Children inherit the affinity; no taskset dependency is needed.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    runner = [args.qemu_user] if args.qemu_user else []
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    rows = []
    log_path = root / "measurements.jsonl"
    with log_path.open("w", buffering=1) as log:
        cases_dir = root / "cases"
        names = {p.stem for p in cases_dir.glob("*.out")}
        names.update(p.name[:-len(".out.sha256")] for p in cases_dir.glob("*.out.sha256"))
        for case in sorted(names):
            if args.cases and case not in args.cases:
                continue
            digest = cases_dir / (case + ".out.sha256")
            expected = digest.read_text().strip() if digest.exists() else (cases_dir / (case + ".out")).read_bytes()
            for repeat in range(args.runs):
                order = args.variants if repeat % 2 == 0 else args.variants[::-1]
                for variant in order:
                    exe = root / variant / (case + ".elf")
                    if not exe.exists():
                        result = {"status": "MISSING"}
                    else:
                        input_path = cases_dir / (case + ".in")
                        if manifest is not None and "in_sha256" not in manifest["cases"].get(case, {}):
                            input_path = None
                        result = run_one(exe, input_path,
                                         expected, args.timeout, runner)
                    row = dict(case=case, variant=variant, repeat=repeat, **result)
                    rows.append(row)
                    line = json.dumps(row)
                    log.write(line + "\n")
                    print(line, flush=True)
    summary = []
    for case in sorted({r["case"] for r in rows}):
        for variant in args.variants:
            group = [r for r in rows if r["case"] == case and r["variant"] == variant]
            passed = len(group) == args.runs and all(r["status"] == "PASS" for r in group)
            entry = {"case": case, "variant": variant, "passed": passed}
            if passed:
                entry.update(median_s=statistics.median(r["time_s"] for r in group),
                             min_s=min(r["time_s"] for r in group),
                             max_s=max(r["time_s"] for r in group),
                             metric=group[0]["metric"])
            summary.append(entry)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if all(r["passed"] for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
