#!/usr/bin/env python3
"""Audit same-file assembly copies and their --pass-stats MIR origins.

Example: python3 tools/benchmark/move_audit.py build/benchmarks/run \
  --variants before after llvm --out results/move-optimization/copies
Counts are static assembly occurrences, not dynamic execution frequencies.
Argument shuffle slots are distinct from register allocator spills.
"""
import argparse
import csv
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
ORIGINS = ('mir', 'entry', 'argument', 'result', 'return', 'address')


def audit(directory):
    rows = []
    for assembly in sorted(directory.glob('*.s')):
        text = assembly.read_text()
        ops = re.findall(r'^\s+([a-z][a-z0-9.]*)\s', text, re.M)
        row = dict(case=assembly.stem, instructions=len(ops), mv=ops.count('mv'),
                   fmv_s=ops.count('fmv.s'), fmv_w_x=ops.count('fmv.w.x'),
                   fmv_x_w=ops.count('fmv.x.w'))
        log = assembly.with_suffix('.compile.log')
        entries = re.findall(r'^backend-copies (.*?): (.*)$', log.read_text(), re.M) if log.exists() else []
        row['origins_available'] = bool(entries)
        keys = [file+'-'+origin for file in ('x', 'f') for origin in ORIGINS]
        keys += ['shuffle-stores', 'shuffle-loads']
        totals = dict.fromkeys(keys, 0)
        for name, fields in entries:
            values = {k: int(v) for k, v in re.findall(r'([a-z-]+)=(\d+)', fields)}
            for key in keys:
                totals[key] += values[key]
        if entries:
            for file, count in [('x', row['mv']), ('f', row['fmv_s'])]:
                if sum(totals[file+'-'+origin] for origin in ORIGINS) != count:
                    raise ValueError(f'{assembly}: {file} copy origins disagree with assembly')
        row.update({key: totals[key] if entries else None for key in keys})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=pathlib.Path)
    parser.add_argument('--variants', nargs='+', default=['before', 'after'])
    parser.add_argument('--out', required=True, type=pathlib.Path, help='Output prefix for CSV and JSON')
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT/'docs' or ROOT/'docs' in out.parents:
        parser.error('comparison results belong in results/ or build/, not docs/')
    rows, totals = [], {}
    for variant in args.variants:
        entries = audit(args.run/variant)
        rows += [dict(variant=variant, **entry) for entry in entries]
        keys = [key for key in entries[0] if key not in ('case', 'origins_available')] if entries else []
        totals[variant] = {key: sum(entry[key] for entry in entries) if
                           all(entry[key] is not None for entry in entries) else None for key in keys}
        totals[variant]['cases'] = len(entries)
        print(variant, json.dumps(totals[variant], sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(dict(totals=totals, cases=rows), indent=2)+'\n')
    if rows:
        with out.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
