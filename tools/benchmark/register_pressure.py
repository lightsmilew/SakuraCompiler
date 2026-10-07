#!/usr/bin/env python3
"""Compare allocator diagnostics and assembly between benchmark variants.

Example: python3 tools/benchmark/register_pressure.py build/benchmarks/run \
  --baseline before --compiler after --out results/pointer-strength-reduction/pressure
Spill counts are static MIR instructions, not dynamic execution counts.
"""
import argparse
import csv
import json
import pathlib
import re

FIELDS = ('peak-x', 'peak-f', 'call-live-x', 'call-live-f', 'rounds',
          'spill-slots', 'spill-loads', 'spill-stores', 'saved-x', 'saved-f')
ROOT = pathlib.Path(__file__).resolve().parents[2]


def diagnostics(directory):
    functions, modules = {}, {}
    for log in sorted(directory.glob('*.compile.log')):
        entries = {}
        for name, fields in re.findall(r'^backend-ra (.*?): (.*)$', log.read_text(), re.M):
            entries[name] = {k: int(v) for k, v in re.findall(r'([a-z-]+)=(\d+)', fields)}
        functions[log.name.removesuffix('.compile.log')] = entries
        assembly = log.with_name(log.name.removesuffix('.compile.log') + '.s')
        ops = re.findall(r'^\s+([a-z][a-z0-9.]*)\s', assembly.read_text(), re.M) if assembly.exists() else []
        modules[log.name.removesuffix('.compile.log')] = dict(instructions=len(ops), moves=ops.count('mv'))
    return functions, modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=pathlib.Path)
    parser.add_argument('--baseline', default='before')
    parser.add_argument('--compiler', default='after')
    parser.add_argument('--out', type=pathlib.Path, required=True, help='Output prefix for .json and .csv')
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT / 'docs' in out.parents:
        parser.error('comparison results belong in results/ or build/, not docs/')
    before, before_asm = diagnostics(args.run / args.baseline)
    after, after_asm = diagnostics(args.run / args.compiler)
    rows = []
    for case in sorted(set(before) | set(after)):
        for name in sorted(set(before.get(case, {})) | set(after.get(case, {}))):
            a, b = before.get(case, {}).get(name), after.get(case, {}).get(name)
            row = dict(case=case, function=name, paired=a is not None and b is not None)
            for field in FIELDS:
                row['before-' + field] = a.get(field) if a else None
                row['after-' + field] = b.get(field) if b else None
                row['delta-' + field] = b[field] - a[field] if a and b and field in a and field in b else None
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(functions=rows, assembly={case: {'before': before_asm.get(case), 'after': after_asm.get(case)}
                  for case in sorted(set(before_asm) | set(after_asm))})
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    if rows:
        with out.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for field in ('spill-slots', 'spill-loads', 'spill-stores'):
        paired = [r for r in rows if r['paired']]
        print(field, sum(r['before-'+field] for r in paired), '->', sum(r['after-'+field] for r in paired))
    regressions = [r for r in rows if r['paired'] and
                   any((r['delta-'+f] or 0) > 0 for f in ('spill-loads', 'spill-stores'))]
    print('functions with increased spill traffic:', len(regressions))
    print('paired functions:', sum(r['paired'] for r in rows),
          '; unpaired functions:', sum(not r['paired'] for r in rows))
    if not any(r['paired'] for r in rows):
        print('No comparable allocator diagnostics; spill changes are unverified.')
    for r in regressions: print(r['case'], r['function'])


if __name__ == '__main__':
    main()
